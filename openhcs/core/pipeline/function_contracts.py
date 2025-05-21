"""
Function-level contract decorators for the pipeline compiler.

This module provides decorators for declaring special input and output contracts
at the function level, enabling compile-time validation of dependencies between
processing functions in the pipeline.

These decorators complement the class-level @special_in and @special_out decorators
by allowing more granular contract declarations at the function level.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Declarative Enforcement
- Clause 246 — Statelessness Mandate
- Clause 251 — Special Output Contract
"""

from typing import Callable, Any, TypeVar, Set, Dict

F = TypeVar('F', bound=Callable[..., Any])


def special_output(key: str) -> Callable[[F], F]:
    """
    Decorator that marks a function as producing a special output.

    This decorator declares that a function produces a special intermediate product
    that can be consumed by other functions using the @special_input decorator.

    Args:
        key: The name of the special output

    Returns:
        The decorated function with special output attribute set
    """
    def decorator(func: F) -> F:
        """Decorator function that sets special output attribute."""
        if not hasattr(func, "__special_outputs__"):
            func.__special_outputs__ = set()
        func.__special_outputs__.add(key)
        return func
    return decorator


def special_input(key: str, required: bool = True) -> Callable[[F], F]:
    """
    Decorator that marks a function as requiring a special input.

    This decorator declares that a function requires a special intermediate product
    that must be produced by another function using the @special_output decorator.

    Args:
        key: The name of the special input
        required: Whether this input is required (default: True)

    Returns:
        The decorated function with special input attribute set
    """
    def decorator(func: F) -> F:
        """Decorator function that sets special input attribute."""
        if not hasattr(func, "__special_inputs__"):
            func.__special_inputs__ = {}
        func.__special_inputs__[key] = required
        return func
    return decorator


def chain_breaker(func: F) -> F:
    """
    Decorator that marks a function as a chain breaker.

    Chain breakers are functions that explicitly break the automatic chaining
    of functions in a pipeline. They are used when a function needs to operate
    independently of the normal pipeline flow.

    Returns:
        The decorated function with chain breaker attribute set
    """
    func.__chain_breaker__ = True
    return func


def get_special_outputs(func: Callable) -> Set[str]:
    """
    Get the special outputs declared by a function.

    Args:
        func: The function to check

    Returns:
        Set of special output keys declared by the function
    """
    return getattr(func, "__special_outputs__", set())


def get_special_inputs(func: Callable) -> Dict[str, bool]:
    """
    Get the special inputs declared by a function.

    Args:
        func: The function to check

    Returns:
        Dictionary mapping special input keys to required flag
    """
    return getattr(func, "__special_inputs__", {})


def is_chain_breaker(func: Callable) -> bool:
    """
    Check if a function is a chain breaker.

    Args:
        func: The function to check

    Returns:
        True if the function is a chain breaker, False otherwise
    """
    return getattr(func, "__chain_breaker__", False)
