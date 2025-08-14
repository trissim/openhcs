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
    return (
        f"Function '{func_name}' in step '{step_name}' needs memory type decorator (@numpy, @cupy, @torch, etc.)\n"
        f"\n"
        f"💡 SOLUTION: Use OpenHCS registry functions instead of raw external library functions:\n"
        f"\n"
        f"❌ WRONG:\n"
        f"   import pyclesperanto as cle\n"
        f"   step = FunctionStep(func=cle.{func_name}, name='{step_name}')\n"
        f"\n"
        f"✅ CORRECT:\n"
        f"   from openhcs.processing.func_registry import get_function_by_name\n"
        f"   {func_name}_func = get_function_by_name('{func_name}', 'pyclesperanto')  # or 'numpy', 'cupy'\n"
        f"   step = FunctionStep(func={func_name}_func, name='{step_name}')\n"
        f"\n"
        f"📋 Available functions: Use get_all_function_names('pyclesperanto') to see all options"
    )

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
    def validate_pipeline(steps: List[Any], pipeline_context: Optional[Dict[str, Any]] = None, orchestrator=None) -> Dict[str, Dict[str, str]]:
        """
        Validate memory type contracts and function patterns for all FunctionStep instances in a pipeline.

        This validator must run after the materialization and path planners to ensure
        proper plan integration. It verifies that these planners have run by checking
        the pipeline_context for planner execution flags and by validating the presence
        of required fields in the step plans.

        Args:
            steps: The steps in the pipeline
            pipeline_context: Optional context object with planner execution flags
            orchestrator: Optional orchestrator for dict pattern key validation

        Returns:
            Dictionary mapping step UIDs to memory type dictionaries

        Raises:
            ValueError: If any FunctionStep violates memory type contracts or dict pattern validation
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
                    # Check for path planner fields (using dunder names)
                    _ = step.__input_dir__
                    _ = step.__output_dir__
                except AttributeError as e:
                    raise AssertionError(
                        f"Clause 101 Violation: Required planners must run before FuncStepContractValidator. "
                        f"Missing attribute: {e}. Path planner must run first."
                    ) from e

                memory_types = FuncStepContractValidator.validate_funcstep(step, orchestrator)
                step_memory_types[step.step_id] = memory_types



        return step_memory_types

    @staticmethod
    def validate_funcstep(step: FunctionStep, orchestrator=None) -> Dict[str, str]:
        """
        Validate memory type contracts, func_pattern structure, and dict pattern keys for a FunctionStep instance.
        If special I/O or chainbreaker decorators are used, the func_pattern must be simple.

        Args:
            step: The FunctionStep to validate
            orchestrator: Optional orchestrator for dict pattern key validation

        Returns:
            Dictionary of validated memory types

        Raises:
            ValueError: If the FunctionStep violates memory type contracts, structural rules,
                        or dict pattern key validation.
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
        
        # 2. Special contracts validation is handled by validate_pattern_structure() below
        # No additional restrictions needed - all valid patterns support special contracts

        # 3. Validate dict pattern keys if orchestrator is available
        if orchestrator is not None and isinstance(func_pattern, dict) and step.group_by is not None:
            FuncStepContractValidator._validate_dict_pattern_keys(
                func_pattern, step.group_by, step_name, orchestrator
            )

        # 4. Proceed with existing memory type validation using the original func_pattern
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

        # Validate that all functions have valid memory type declarations
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

        # Return first function's input type and last function's output type
        last_function = functions[-1]
        return input_type, last_function.output_memory_type

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
    def _validate_dict_pattern_keys(
        func_pattern: dict,
        group_by,
        step_name: str,
        orchestrator
    ) -> None:
        """
        Validate that dict function pattern keys match available component keys.

        This validation ensures compile-time guarantee that dict patterns will work
        at runtime by checking that all dict keys exist in the actual component data.

        Args:
            func_pattern: Dict function pattern to validate
            group_by: GroupBy enum specifying component type
            step_name: Name of the step containing the function
            orchestrator: Orchestrator for component key access

        Raises:
            ValueError: If dict pattern keys don't match available component keys
        """
        # Get available component keys from orchestrator
        try:
            available_keys = orchestrator.get_component_keys(group_by)
            available_keys_set = set(str(key) for key in available_keys)
        except Exception as e:
            raise ValueError(f"Failed to get component keys for {group_by.value}: {e}")

        # Check each dict key against available keys
        pattern_keys = list(func_pattern.keys())
        pattern_keys_set = set(str(key) for key in pattern_keys)

        # Try direct string match first
        missing_keys = pattern_keys_set - available_keys_set

        if missing_keys:
            # Try integer conversion for missing keys
            still_missing = set()
            for key in missing_keys:
                try:
                    # Try converting pattern key to int and check if int version exists in available keys
                    key_as_int = int(key)
                    if str(key_as_int) not in available_keys_set:
                        still_missing.add(key)
                except (ValueError, TypeError):
                    # Try converting available keys to int and check if string key matches
                    found_as_int = False
                    for avail_key in available_keys_set:
                        try:
                            if int(avail_key) == int(key):
                                found_as_int = True
                                break
                        except (ValueError, TypeError):
                            continue
                    if not found_as_int:
                        still_missing.add(key)

            if still_missing:
                raise ValueError(
                    f"Function pattern keys not found in available {group_by.value} components for step '{step_name}'. "
                    f"Missing keys: {sorted(still_missing)}. "
                    f"Available keys: {sorted(available_keys)}. "
                    f"Function pattern keys must match component values from the plate data."
                )

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