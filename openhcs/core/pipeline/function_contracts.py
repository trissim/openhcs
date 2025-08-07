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

def special_outputs(*output_specs) -> Callable[[F], F]:
    """
    Decorator that marks a function as producing special outputs.

    Args:
        *output_specs: Either strings or (string, materialization_function) tuples
                      - String only: "positions" - no materialization function
                      - Tuple: ("cell_counts", materialize_cell_counts) - with materialization

    Examples:
        @special_outputs("positions", "metadata")  # String only
        def process_image(image):
            return processed_image, positions, metadata

        @special_outputs(("cell_counts", materialize_cell_counts))  # With materialization
        def count_cells(image):
            return processed_image, cell_count_results

        @special_outputs("positions", ("cell_counts", materialize_cell_counts))  # Mixed
        def analyze_image(image):
            return processed_image, positions, cell_count_results
    """
    def decorator(func: F) -> F:
        special_outputs_info = {}
        output_keys = set()

        for spec in output_specs:
            if isinstance(spec, str):
                # String only - no materialization function
                output_keys.add(spec)
                special_outputs_info[spec] = None
            elif isinstance(spec, tuple) and len(spec) == 2:
                # (key, materialization_function) tuple
                key, mat_func = spec
                if not isinstance(key, str):
                    raise ValueError(f"Special output key must be string, got {type(key)}: {key}")
                if not callable(mat_func):
                    raise ValueError(f"Materialization function must be callable, got {type(mat_func)}: {mat_func}")
                output_keys.add(key)
                special_outputs_info[key] = mat_func
            else:
                raise ValueError(f"Invalid special output spec: {spec}. Must be string or (string, function) tuple.")

        # Set both attributes for backward compatibility and new functionality
        func.__special_outputs__ = output_keys  # For path planner (backward compatibility)
        func.__materialization_functions__ = special_outputs_info  # For materialization system
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





