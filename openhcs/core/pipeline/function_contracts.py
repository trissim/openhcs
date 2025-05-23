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


# Old special_output and special_input decorators are removed.

def special_outputs(*output_names: str) -> Callable[[F], F]:
    """
    Decorator that marks a function as producing special outputs.

    Args:
        *output_names: Names of the additional return values (excluding the first)
                      that can be consumed by other functions

    Example:
        @special_outputs("positions", "metadata")
        def process_image(image):
            # First return value is always the processed image (3D array)
            # Second return value is positions
            # Third return value is metadata
            return processed_image, positions, metadata
    """
    def decorator(func: F) -> F:
        func.__special_outputs__ = set(output_names)
        return func
    return decorator


def special_inputs(*input_names: str) -> Callable[[F], F]:
    """
    Decorator that marks a function as requiring special inputs.

    Args:
        *input_names: Names of the additional input parameters (excluding the first)
                     that must be produced by other functions

    Example:
        @special_inputs("positions", "metadata")
        def stitch_images(image_stack, positions, metadata):
            # First parameter is always the input image (3D array)
            # Additional parameters are special inputs from other functions
            return stitched_image
    """
    def decorator(func: F) -> F:
        # For special_inputs, we store them as a dictionary with True as the value,
        # similar to the old special_input decorator, for compatibility with
        # existing logic in PathPlanner that expects a dict.
        # The 'required' flag is implicitly True for all named inputs here.
        # If optional special inputs are needed later, this structure can be extended.
        func.__special_inputs__ = {name: True for name in input_names}
        return func
    return decorator


def chain_breaker(func: F) -> F:
    """
    Decorator that marks a function as a chain breaker.

    Chain breakers are functions that explicitly break the automatic chaining
    of functions in a pipeline. They are used when a function needs to operate
    independently of the normal pipeline flow.

    The path planner will force any step following a chain breaker step
    (a step with a single function in the pattern) to have its input be the
    same as the input of the step at the beginning of the pipeline.

    This decorator takes no arguments - its presence alone is sufficient to
    mark a function as a chain breaker.

    Returns:
        The decorated function with chain breaker attribute set
    """
    func.__chain_breaker__ = True
    return func


