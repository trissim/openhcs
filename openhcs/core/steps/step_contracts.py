"""
Step contract decorators for OpenHCS.

This module provides decorators for declaring special input and output contracts
between steps in a pipeline. These decorators allow steps to declare non-standard
intermediate products that are produced by one step and consumed by another.

Note: The validation of these contracts is handled by the path planner during
pipeline compilation, not by this module.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Declarative Enforcement
- Clause 246 — Statelessness Mandate
- Clause 251 — Special Output Contract
- Clause 631 — Pipeline Path Connectivity
"""

import logging
from typing import Any, Callable, Dict, Set, Type, TypeVar

logger = logging.getLogger(__name__)

# Type variable for step classes
S = TypeVar('S', bound=Type[Any])

# Global registry of special contracts
# Maps step class to its special inputs and outputs
SPECIAL_CONTRACT_REGISTRY: Dict[Type, Dict[str, Set[str]]] = {}

# Error messages for decorator validation
ERROR_DUPLICATE_SPECIAL_OUT = (
    "Clause 251 Violation: Step class '{}' already declares special_out '{}'"
)
ERROR_DUPLICATE_SPECIAL_IN = (
    "Clause 251 Violation: Step class '{}' already declares special_in '{}'"
)


def special_out(key: str) -> Callable[[S], S]:
    """
    Decorator that marks a step class as producing a special output.

    This decorator declares that a step produces a special intermediate product
    that can be consumed by other steps using the @special_in decorator.

    Args:
        key: The name of the special output

    Returns:
        The decorated step class with special output attribute set

    Raises:
        ValueError: If the step class already declares the same special output
    """
    def decorator(step_cls: S) -> S:
        """Decorator function that sets special output attribute."""
        # Initialize registry entry for this step class if it doesn't exist
        if step_cls not in SPECIAL_CONTRACT_REGISTRY:
            SPECIAL_CONTRACT_REGISTRY[step_cls] = {
                'special_out': set(),
                'special_in': set()
            }

        # Check if the step class already declares this special output
        if key in SPECIAL_CONTRACT_REGISTRY[step_cls]['special_out']:
            raise ValueError(ERROR_DUPLICATE_SPECIAL_OUT.format(step_cls.__name__, key))

        # Add the special output to the registry
        SPECIAL_CONTRACT_REGISTRY[step_cls]['special_out'].add(key)

        # Add special_outputs attribute to the step class if it doesn't exist
        if not hasattr(step_cls, 'special_outputs'):
            step_cls.special_outputs = set()

        # Add the key to the step class's special_outputs set
        step_cls.special_outputs.add(key)

        # Return the class unchanged (no wrapper)
        return step_cls

    return decorator


def special_in(key: str, required: bool = True) -> Callable[[S], S]:
    """
    Decorator that marks a step class as requiring a special input.

    This decorator declares that a step requires a special intermediate product
    that must be produced by another step using the @special_out decorator.

    Args:
        key: The name of the special input
        required: Whether this input is required (default: True)

    Returns:
        The decorated step class with special input attribute set

    Raises:
        ValueError: If the step class already declares the same special input
    """
    def decorator(step_cls: S) -> S:
        """Decorator function that sets special input attribute."""
        # Initialize registry entry for this step class if it doesn't exist
        if step_cls not in SPECIAL_CONTRACT_REGISTRY:
            SPECIAL_CONTRACT_REGISTRY[step_cls] = {
                'special_out': set(),
                'special_in': set()
            }

        # Check if the step class already declares this special input
        if key in SPECIAL_CONTRACT_REGISTRY[step_cls]['special_in']:
            raise ValueError(ERROR_DUPLICATE_SPECIAL_IN.format(step_cls.__name__, key))

        # Add the special input to the registry
        SPECIAL_CONTRACT_REGISTRY[step_cls]['special_in'].add(key)

        # Add special_inputs attribute to the step class if it doesn't exist
        if not hasattr(step_cls, 'special_inputs'):
            step_cls.special_inputs = {}

        # Add the key to the step class's special_inputs dict with required flag
        step_cls.special_inputs[key] = required

        # Return the class unchanged (no wrapper)
        return step_cls

    return decorator


# Validation of special contracts is now handled by the path planner


def get_declared_special_outputs(step_cls: Type) -> Set[str]:
    """
    Get the special outputs declared by a step class.

    Args:
        step_cls: The step class to check

    Returns:
        Set of special output keys declared by the step class
    """
    has_outputs = (step_cls in SPECIAL_CONTRACT_REGISTRY and
                  'special_out' in SPECIAL_CONTRACT_REGISTRY[step_cls])
    if has_outputs:
        return SPECIAL_CONTRACT_REGISTRY[step_cls]['special_out']
    return set()


def get_declared_special_inputs(step_cls: Type) -> Dict[str, bool]:
    """
    Get the special inputs declared by a step class.

    Args:
        step_cls: The step class to check

    Returns:
        Dictionary mapping special input keys to required flag
    """
    result = {}
    has_inputs = (step_cls in SPECIAL_CONTRACT_REGISTRY and
                 'special_in' in SPECIAL_CONTRACT_REGISTRY[step_cls])
    if has_inputs:
        for key in SPECIAL_CONTRACT_REGISTRY[step_cls]['special_in']:
            required = True
            if hasattr(step_cls, 'special_inputs'):
                required = step_cls.special_inputs.get(key, True)
            result[key] = required
    return result
