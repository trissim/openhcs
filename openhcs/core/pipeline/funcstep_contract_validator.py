"""
FuncStep memory contract validator for OpenHCS.

This module provides the FuncStepContractValidator class, which is responsible for
validating memory type declarations for FunctionStep instances in a pipeline.

Doctrinal Clauses:
- Clause 65 — No Fallback Logic
- Clause 88 — No Inferred Capabilities
- Clause 101 — Memory Type Declaration
- Clause 106-A — Declared Memory Types
- Clause 308 — Named Positional Enforcement
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from openhcs.constants.constants import VALID_MEMORY_TYPES
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)

# ===== DECLARATIVE DEFAULT VALUES =====
# These declarations control defaults and may be moved to configuration in the future

# Default error messages for validation failures
ERROR_MISSING_MEMORY_TYPE = (
    "Clause 101 Violation: Function '{0}' in step '{1}' does not have explicit "
    "memory type declarations. Use @memory_types, @numpy, @cupy, @torch, or @tensorflow decorators."
)

ERROR_INCONSISTENT_MEMORY_TYPES = (
    "Clause 101 Violation: Functions in step '{0}' have inconsistent memory types. "
    "Function '{1}' has input_memory_type='{2}', output_memory_type='{3}', but function '{4}' "
    "has input_memory_type='{5}', output_memory_type='{6}'."
)

ERROR_INVALID_MEMORY_TYPE = (
    "Clause 101 Violation: Unknown memory type in function '{0}': input='{1}', output='{2}'. "
    "Valid memory types are: {3}."
)

ERROR_INVALID_FUNCTION = (
    "Clause 101 Violation: Invalid function in {0}: {1}. "
    "All functions must be callable objects with memory type declarations."
)

ERROR_INVALID_PATTERN = (
    "Clause 101 Violation: Invalid function pattern: {0}. "
    "Must be a callable, tuple of (callable, kwargs), list of callables, or dict of callables."
)

ERROR_MISSING_REQUIRED_ARGS = (
    "Clause 308 Violation: Function '{0}' in step '{1}' is missing required positional arguments in kwargs: {2}. "
    "All required positional arguments must be explicitly provided in the kwargs dict when using (func, kwargs) pattern."
)

class FuncStepContractValidator:
    """
    Validator for FunctionStep memory type contracts.

    This validator enforces Clause 101 (Memory Type Declaration), Clause 88
    (No Inferred Capabilities), and Clause 308 (Named Positional Enforcement)
    by requiring explicit memory type declarations and named positional arguments
    for all FunctionStep instances and their functions.

    Key principles:
    1. All functions in a FunctionStep must have consistent memory types
    2. The shared memory types are set as the step's memory types in the step plan
    3. Memory types must be validated at plan time, not runtime
    4. No fallback or inference of memory types is allowed
    5. All function patterns (callable, tuple, list, dict) are supported
    6. When using (func, kwargs) pattern, all required positional arguments must be
       explicitly provided in the kwargs dict
    """

    @staticmethod
    def validate_pipeline(steps: List[Any], pipeline_context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, str]]:
        """
        Validate memory type contracts for all FunctionStep instances in a pipeline.

        This validator must run after the materialization and path planners to ensure
        proper plan integration. It verifies that these planners have run by checking
        the pipeline_context for planner execution flags and by validating the presence
        of required fields in the step plans.

        Args:
            steps: The steps in the pipeline
            pipeline_context: Optional context object with planner execution flags

        Returns:
            Dictionary mapping step UIDs to memory type dictionaries

        Raises:
            ValueError: If any FunctionStep violates memory type contracts
            AssertionError: If required planners have not run before this validator
        """
        # Validate steps
        if not steps:
            logger.warning("No steps provided to FuncStepContractValidator")
            return {}

        # Verify that required planners have run before this validator
        if pipeline_context is not None:
            # Check for planner execution flags in the context
            if not pipeline_context.get("path_planner_done", False):
                raise AssertionError(
                    "Clause 101 Violation: Path planner must run before FuncStepContractValidator. "
                    "Set pipeline_context['path_planner_done'] = True after running the path planner."
                )

            if not pipeline_context.get("materialization_planner_done", False):
                raise AssertionError(
                    "Clause 101 Violation: Materialization planner must run before FuncStepContractValidator. "
                    "Set pipeline_context['materialization_planner_done'] = True after running the materialization planner."
                )
        else:
            logger.warning(
                "No pipeline_context provided to FuncStepContractValidator. "
                "Cannot verify planner execution order. Falling back to attribute checks."
            )

        # Create step memory types dictionary
        step_memory_types = {}

        # Process each step in the pipeline
        for step in steps:
            # Only validate FunctionStep instances
            if isinstance(step, FunctionStep):
                # Verify that other planners have run before this validator by checking attributes
                # This is a fallback verification when pipeline_context is not provided
                try:
                    # Check for materialization planner fields
                    _ = step.requires_disk_input
                    _ = step.requires_disk_output

                    # Check for path planner fields
                    _ = step.input_dir
                    _ = step.output_dir
                except AttributeError as e:
                    raise AssertionError(
                        f"Clause 101 Violation: Required planners must run before FuncStepContractValidator. "
                        f"Missing attribute: {e}. Materialization and path planners must run first."
                    ) from e

                memory_types = FuncStepContractValidator.validate_funcstep(step)
                step_memory_types[step.uid] = memory_types

        return step_memory_types

    @staticmethod
    def validate_funcstep(step: FunctionStep) -> Dict[str, str]:
        """
        Validate memory type contracts for a FunctionStep instance.

        Args:
            step: The FunctionStep to validate

        Returns:
            Dictionary of validated memory types

        Raises:
            ValueError: If the FunctionStep violates memory type contracts
        """
        # Extract the function pattern from the step
        func = step.func

        # Validate the function pattern and get the shared memory types
        input_type, output_type = FuncStepContractValidator.validate_function_pattern(
            func, step.name)

        # Return the validated memory types
        return {
            'input_memory_type': input_type,
            'output_memory_type': output_type
        }

    @staticmethod
    def validate_function_pattern(
        func: Any,
        step_name: str
    ) -> Tuple[str, str]:
        """
        Validate memory type contracts for a function pattern.

        Args:
            func: The function pattern to validate
            step_name: The name of the step containing the function

        Returns:
            Tuple of (input_memory_type, output_memory_type)

        Raises:
            ValueError: If the function pattern violates memory type contracts
        """
        # Extract all functions from the pattern
        functions = FuncStepContractValidator.validate_pattern_structure(func, step_name)

        if not functions:
            raise ValueError(f"No valid functions found in pattern for step {step_name}")

        # Get memory types from the first function
        first_fn = functions[0]

        # Validate that the function has explicit memory type declarations
        try:
            input_type = first_fn.input_memory_type
            output_type = first_fn.output_memory_type
        except AttributeError as exc:
            raise ValueError(ERROR_MISSING_MEMORY_TYPE.format(
                first_fn.__name__, step_name
            )) from exc

        # Validate memory types against known valid types
        if input_type not in VALID_MEMORY_TYPES or output_type not in VALID_MEMORY_TYPES:
            raise ValueError(ERROR_INVALID_MEMORY_TYPE.format(
                first_fn.__name__,
                input_type,
                output_type,
                ", ".join(sorted(VALID_MEMORY_TYPES))
            ))

        # Validate that all functions have the same memory types
        for fn in functions[1:]:
            # Validate that the function has explicit memory type declarations
            try:
                fn_input_type = fn.input_memory_type
                fn_output_type = fn.output_memory_type
            except AttributeError as exc:
                raise ValueError(ERROR_MISSING_MEMORY_TYPE.format(
                    fn.__name__, step_name
                )) from exc

            # Validate memory types against known valid types
            if fn_input_type not in VALID_MEMORY_TYPES or fn_output_type not in VALID_MEMORY_TYPES:
                raise ValueError(ERROR_INVALID_MEMORY_TYPE.format(
                    fn.__name__,
                    fn_input_type,
                    fn_output_type,
                    ", ".join(sorted(VALID_MEMORY_TYPES))
                ))

            # Validate that the function's memory types match the first function's memory types
            if fn_input_type != input_type or fn_output_type != output_type:
                raise ValueError(ERROR_INCONSISTENT_MEMORY_TYPES.format(
                    step_name,
                    first_fn.__name__,
                    input_type,
                    output_type,
                    fn.__name__,
                    fn_input_type,
                    fn_output_type
                ))

        # Return the shared memory types
        return input_type, output_type

    @staticmethod
    def _validate_required_args(func: Callable, kwargs: Dict[str, Any], step_name: str) -> None:
        """
        Validate that all required positional arguments are provided in kwargs.

        This enforces Clause 308 (Named Positional Enforcement) by requiring that
        all required positional arguments are explicitly provided in the kwargs dict
        when using the (func, kwargs) pattern.

        Args:
            func: The function to validate
            kwargs: The kwargs dict to check
            step_name: The name of the step containing the function

        Raises:
            ValueError: If any required positional arguments are missing from kwargs
        """
        # Get the function signature
        sig = inspect.signature(func)

        # Collect names of required positional arguments
        required_args = []
        for name, param in sig.parameters.items():
            # Check if parameter is positional (POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD)
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                # Check if parameter has no default value
                if param.default is inspect.Parameter.empty:
                    required_args.append(name)

        # Check if all required args are in kwargs
        missing_args = [arg for arg in required_args if arg not in kwargs]

        # Raise error if any required args are missing
        if missing_args:
            raise ValueError(ERROR_MISSING_REQUIRED_ARGS.format(
                func.__name__, step_name, ", ".join(missing_args)
            ))

    @staticmethod
    def validate_pattern_structure(
        func: Any,
        step_name: str
    ) -> List[Callable]:
        """
        Validate and extract all functions from a function pattern.

        This is a public wrapper for _extract_functions_from_pattern that provides
        a stable API for pattern structure validation.

        Supports nested patterns of arbitrary depth, including:
        - Direct callable
        - Tuple of (callable, kwargs)
        - List of callables or patterns
        - Dict of keyed callables or patterns

        Args:
            func: The function pattern to validate and extract functions from
            step_name: The name of the step or component containing the function

        Returns:
            List of functions in the pattern

        Raises:
            ValueError: If the function pattern is invalid
        """
        return FuncStepContractValidator._extract_functions_from_pattern(func, step_name)

    @staticmethod
    def _extract_functions_from_pattern(
        func: Any,
        step_name: str
    ) -> List[Callable]:
        """
        Extract all functions from a function pattern.

        Supports nested patterns of arbitrary depth, including:
        - Direct callable
        - Tuple of (callable, kwargs)
        - List of callables or patterns
        - Dict of keyed callables or patterns

        Args:
            func: The function pattern to extract functions from
            step_name: The name of the step containing the function

        Returns:
            List of functions in the pattern

        Raises:
            ValueError: If the function pattern is invalid
        """
        functions = []

        # Case 1: Direct callable
        if callable(func) and not isinstance(func, type):
            functions.append(func)
            return functions

        # Case 2: Tuple of (callable, kwargs)
        if (isinstance(func, tuple) and len(func) == 2 and
                callable(func[0]) and isinstance(func[1], dict)):
            # Validate that all required positional arguments are provided in kwargs
            # This enforces Clause 308 (Named Positional Enforcement)
            FuncStepContractValidator._validate_required_args(func[0], func[1], step_name)

            functions.append(func[0])
            return functions

        # Case 3: List of patterns
        if isinstance(func, list):
            for i, f in enumerate(func):
                # Recursively extract functions from nested patterns
                if isinstance(f, (list, dict, tuple)) or (callable(f) and not isinstance(f, type)):
                    nested_functions = FuncStepContractValidator._extract_functions_from_pattern(
                        f, step_name)
                    functions.extend(nested_functions)
                else:
                    raise ValueError(ERROR_INVALID_FUNCTION.format(
                        f"list at index {i}", f
                    ))
            return functions

        # Case 4: Dict of keyed patterns
        if isinstance(func, dict):
            for key, f in func.items():
                # Recursively extract functions from nested patterns
                if isinstance(f, (list, dict, tuple)) or (callable(f) and not isinstance(f, type)):
                    nested_functions = FuncStepContractValidator._extract_functions_from_pattern(
                        f, step_name)
                    functions.extend(nested_functions)
                else:
                    raise ValueError(ERROR_INVALID_FUNCTION.format(
                        f"dict with key '{key}'", f
                    ))
            return functions

        # Invalid type
        raise ValueError(ERROR_INVALID_PATTERN.format(func))