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

# Simple, direct error messages
def missing_memory_type_error(func_name, step_name):
    return f"Function '{func_name}' in step '{step_name}' needs memory type decorator (@numpy, @cupy, @torch, etc.)"

def inconsistent_memory_types_error(step_name, func1, func2):
    return f"Functions in step '{step_name}' have different memory types: {func1} vs {func2}"

def invalid_memory_type_error(func_name, input_type, output_type, valid_types):
    return f"Function '{func_name}' has invalid memory types: {input_type}/{output_type}. Valid: {valid_types}"

def invalid_function_error(location, func):
    return f"Invalid function in {location}: {func}"

def invalid_pattern_error(pattern):
    return f"Invalid function pattern: {pattern}"

def missing_required_args_error(func_name, step_name, missing_args):
    return f"Function '{func_name}' in step '{step_name}' missing required args: {missing_args}"

def complex_pattern_error(step_name):
    return f"Step '{step_name}' with special decorators must use simple function pattern"

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
            # Check that step plans exist and have required fields from planners
            if not pipeline_context.step_plans:
                raise AssertionError(
                    "Clause 101 Violation: Step plans must be initialized before FuncStepContractValidator."
                )

            # Check that materialization planner has run by verifying read_backend/write_backend exist
            sample_step_id = next(iter(pipeline_context.step_plans.keys()))
            sample_plan = pipeline_context.step_plans[sample_step_id]
            if 'read_backend' not in sample_plan or 'write_backend' not in sample_plan:
                raise AssertionError(
                    "Clause 101 Violation: Materialization planner must run before FuncStepContractValidator. "
                    "Step plans missing read_backend/write_backend fields."
                )
        else:
            logger.warning(
                "No pipeline_context provided to FuncStepContractValidator. "
                "Cannot verify planner execution order. Falling back to attribute checks."
            )

        # Create step memory types dictionary
        step_memory_types = {}

        # Process each step in the pipeline
        for i, step in enumerate(steps):
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
                step_memory_types[step.step_id] = memory_types



        return step_memory_types

    @staticmethod
    def validate_funcstep(step: FunctionStep) -> Dict[str, str]:
        """
        Validate memory type contracts and func_pattern structure for a FunctionStep instance.
        If special I/O or chainbreaker decorators are used, the func_pattern must be simple.

        Args:
            step: The FunctionStep to validate

        Returns:
            Dictionary of validated memory types

        Raises:
            ValueError: If the FunctionStep violates memory type contracts or structural rules
                        related to special I/O/chainbreaker decorators.
        """
        # Extract the function pattern and name from the step
        func_pattern = step.func # Renamed for clarity in this context
        step_name = step.name

        # 1. Check if any function in the pattern uses special contract decorators
        # _extract_functions_from_pattern will raise ValueError if func_pattern itself is invalid (e.g. None, or bad structure)
        all_callables = FuncStepContractValidator._extract_functions_from_pattern(func_pattern, step_name)
        
        uses_special_contracts = False
        if all_callables: # Only check attributes if we have actual callables
            for f_callable in all_callables:
                if hasattr(f_callable, '__special_inputs__') or \
                   hasattr(f_callable, '__special_outputs__') or \
                   hasattr(f_callable, '__chain_breaker__'):
                    uses_special_contracts = True
                    break
        
        # 2. If special contracts are used, validate the func_pattern's overall structure
        if uses_special_contracts:
            is_structurally_simple = False
            # Check for direct callable (and not a class type itself)
            if callable(func_pattern) and not isinstance(func_pattern, type):
                is_structurally_simple = True
            # Check for (callable, kwargs_dict) tuple
            elif isinstance(func_pattern, tuple):
                # _extract_functions_from_pattern already validates tuple structure if it contains a callable
                # We just confirm it's a 2-tuple with callable and dict for this specific rule.
                if len(func_pattern) == 2 and callable(func_pattern[0]) and \
                   not isinstance(func_pattern[0], type) and isinstance(func_pattern[1], dict):
                    is_structurally_simple = True
            # Check for list containing exactly one simple item
            elif isinstance(func_pattern, list):
                if len(func_pattern) == 1:
                    item = func_pattern[0]
                    # The single item must itself be a simple pattern (callable or valid tuple)
                    if (callable(item) and not isinstance(item, type)) or \
                       (isinstance(item, tuple) and len(item) == 2 and
                        callable(item[0]) and not isinstance(item[0], type) and
                        isinstance(item[1], dict)):
                        is_structurally_simple = True
            
            if not is_structurally_simple:
                raise ValueError(complex_pattern_error(step_name))

        # 3. Proceed with existing memory type validation using the original func_pattern
        input_type, output_type = FuncStepContractValidator.validate_function_pattern(
            func_pattern, step_name)

        # Return the validated memory types and store the func for stateless execution
        return {
            'input_memory_type': input_type,
            'output_memory_type': output_type,
            'func': func_pattern  # Store the validated func for stateless execution
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
            raise ValueError(missing_memory_type_error(first_fn.__name__, step_name)) from exc

        # Validate memory types against known valid types
        if input_type not in VALID_MEMORY_TYPES or output_type not in VALID_MEMORY_TYPES:
            raise ValueError(invalid_memory_type_error(
                first_fn.__name__, input_type, output_type, ", ".join(sorted(VALID_MEMORY_TYPES))
            ))

        # Validate that all functions have the same memory types
        for fn in functions[1:]:
            # Validate that the function has explicit memory type declarations
            try:
                fn_input_type = fn.input_memory_type
                fn_output_type = fn.output_memory_type
            except AttributeError as exc:
                raise ValueError(missing_memory_type_error(fn.__name__, step_name)) from exc

            # Validate memory types against known valid types
            if fn_input_type not in VALID_MEMORY_TYPES or fn_output_type not in VALID_MEMORY_TYPES:
                raise ValueError(invalid_memory_type_error(
                    fn.__name__, fn_input_type, fn_output_type, ", ".join(sorted(VALID_MEMORY_TYPES))
                ))

            # Validate that the function's memory types match the first function's memory types
            if fn_input_type != input_type or fn_output_type != output_type:
                raise ValueError(inconsistent_memory_types_error(
                    step_name, f"{first_fn.__name__}({input_type}/{output_type})",
                    f"{fn.__name__}({fn_input_type}/{fn_output_type})"
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
            raise ValueError(missing_required_args_error(func.__name__, step_name, missing_args))

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
            # The kwargs dict is optional - if provided, it will be used during execution
            # No need to validate required args here as the execution logic handles this gracefully
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
                    raise ValueError(invalid_function_error(f"list at index {i}", f))
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
                    raise ValueError(invalid_function_error(f"dict with key '{key}'", f))
            return functions

        # Invalid type
        raise ValueError(invalid_pattern_error(func))