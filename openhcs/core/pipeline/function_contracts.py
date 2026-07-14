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

from typing import Callable, Any, TypeVar

from openhcs.processing.materialization import MaterializationSpec

F = TypeVar('F', bound=Callable[..., Any])


# Old special_output and special_input decorators are removed.

def special_outputs(*output_specs) -> Callable[[F], F]:
    """
    Decorator that marks a function as producing special outputs.

    Args:
        *output_specs: Either strings or (string, MaterializationSpec) tuples
                      - String only: "positions" - no materialization
                      - Tuple: ("cell_counts", MaterializationSpec(CsvOptions(...))) - writer-based materialization

    Examples:
        @special_outputs("positions", "metadata")  # String only
        def process_image(image):
            return processed_image, positions, metadata

        @special_outputs(("cell_counts", MaterializationSpec(CsvOptions(...))))  # With materialization spec
        def count_cells(image):
            return processed_image, cell_count_results

        @special_outputs("positions", ("cell_counts", MaterializationSpec(CsvOptions(...))))  # Mixed
        def analyze_image(image):
            return processed_image, positions, cell_count_results
    """
    def decorator(func: F) -> F:
        materialization_specs = {}
        output_keys = []

        for spec in output_specs:
            if isinstance(spec, str):
                # String only - no materialization function
                output_keys.append(spec)
            elif isinstance(spec, tuple) and len(spec) == 2:
                # (key, MaterializationSpec) tuple or registered materializer callable
                key, mat_spec = spec
                if not isinstance(key, str):
                    raise ValueError(f"Special output key must be string, got {type(key)}: {key}")
                if not isinstance(mat_spec, MaterializationSpec):
                    raise ValueError(
                        "Materialization spec must be a MaterializationSpec. "
                        f"Got {type(mat_spec)} for key '{key}'."
                    )
                output_keys.append(key)
                materialization_specs[key] = mat_spec
            else:
                raise ValueError(
                    f"Invalid special output spec: {spec}. "
                    "Must be string or (string, MaterializationSpec) tuple."
                )

        # Set both attributes for backward compatibility and new functionality
        # Return values are matched to these keys positionally, so declaration
        # order is part of the function contract and must not be discarded.
        func.__special_outputs__ = tuple(output_keys)
        func.__materialization_specs__ = materialization_specs  # For materialization system
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
