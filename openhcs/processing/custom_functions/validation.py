"""
Code validation utilities for custom functions.

Provides validation before exec() to catch common errors early and provide
helpful error messages. Follows OpenHCS fail-loud principle: invalid code
should fail immediately with clear diagnostics.

Security note: This module provides basic validation but does not sandbox
exec(). Custom functions execute with full Python privileges.
"""

import ast
import inspect
from dataclasses import dataclass
from typing import Callable

from arraybridge import MemoryContractAttribute, MemoryType

from openhcs.core.callable_contract import CallableMetadata


MEMORY_DECORATOR_NAMES = frozenset(memory_type.value for memory_type in MemoryType)


class ValidationError(Exception):
    """
    Exception raised when custom function code is invalid.

    Attributes:
        message: Human-readable error description
        line_number: Optional line number where error occurred
        code_snippet: Optional code snippet showing the error
    """

    def __init__(self, message: str, line_number: int = 0, code_snippet: str = ""):
        self.message = message
        self.line_number = line_number
        self.code_snippet = code_snippet
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with line number and code snippet if available."""
        parts = [self.message]
        if self.line_number > 0:
            parts.append(f"Line {self.line_number}")
        if self.code_snippet:
            parts.append(f"Code: {self.code_snippet}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of code validation.

    Attributes:
        is_valid: Whether the code passed validation
        errors: List of validation error messages
        warnings: List of non-fatal warning messages
        function_names: List of function names found in the code
    """

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    function_names: list[str]


def validate_syntax(code: str) -> ValidationResult:
    """
    Validate Python syntax using ast.parse.

    Args:
        code: Python code string to validate

    Returns:
        ValidationResult with syntax validation results
    """
    try:
        ast.parse(code)
        return ValidationResult(
            is_valid=True, errors=[], warnings=[], function_names=[]
        )
    except SyntaxError as e:
        error_msg = f"Syntax error: {e.msg}"
        return ValidationResult(
            is_valid=False, errors=[error_msg], warnings=[], function_names=[]
        )


def validate_imports(code: str) -> ValidationResult:
    """
    Validate that code doesn't import dangerous modules.

    Args:
        code: Python code string to validate

    Returns:
        ValidationResult with import validation results
    """
    # Modules that should not be imported in custom functions
    dangerous_modules: set[str] = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "glob",
        "socket",
        "urllib",
        "requests",
        "http",
        "eval",
        "exec",
        "compile",
        "__import__",
    }

    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors will be caught by validate_syntax
        return ValidationResult(
            is_valid=True, errors=[], warnings=[], function_names=[]
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in dangerous_modules:
                    errors.append(
                        f"Dangerous import detected: '{alias.name}'. "
                        f"Module '{module_name}' is not allowed in custom functions."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in dangerous_modules:
                    errors.append(
                        f"Dangerous import detected: 'from {node.module}'. "
                        f"Module '{module_name}' is not allowed in custom functions."
                    )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid, errors=errors, warnings=warnings, function_names=[]
    )


def validate_decorator(code: str) -> ValidationResult:
    """
    Validate that at least one function has a memory type decorator.

    Args:
        code: Python code string to validate

    Returns:
        ValidationResult with decorator validation results
    """
    errors: list[str] = []
    warnings: list[str] = []
    function_names: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors will be caught by validate_syntax
        return ValidationResult(
            is_valid=True, errors=[], warnings=[], function_names=[]
        )

    # Find all function definitions
    functions_with_decorators: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)

            # Check if function has memory type decorator
            has_memory_decorator = False
            for decorator in node.decorator_list:
                # Handle simple decorator: @numpy
                if isinstance(decorator, ast.Name):
                    if decorator.id in MEMORY_DECORATOR_NAMES:
                        has_memory_decorator = True
                        functions_with_decorators.append(node.name)
                        break

                # Handle attribute decorator: @decorators.numpy
                elif isinstance(decorator, ast.Attribute):
                    if decorator.attr in MEMORY_DECORATOR_NAMES:
                        has_memory_decorator = True
                        functions_with_decorators.append(node.name)
                        break

            if not has_memory_decorator and not node.name.startswith("_"):
                warnings.append(
                    f"Function '{node.name}' lacks memory type decorator. "
                    "Must be decorated with one of: "
                    + ", ".join(f"@{name}" for name in MEMORY_DECORATOR_NAMES)
                )

    if not functions_with_decorators:
        errors.append(
            "No valid functions found with memory type decorators. "
            "Functions must be decorated with one of: "
            + ", ".join(f"@{name}" for name in MEMORY_DECORATOR_NAMES)
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        function_names=function_names,
    )


def validate_function_signature(func: Callable) -> ValidationResult:
    """
    Validate that a function has correct signature (first param is 'image').

    Args:
        func: Function to validate

    Returns:
        ValidationResult with signature validation results
    """
    errors: list[str] = []
    warnings: list[str] = []

    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    if not params:
        errors.append(
            f"Function '{func.__name__}' has no parameters. "
            "First parameter must be 'image' (3D array: C, Y, X)."
        )
    elif params[0] != "image":
        errors.append(
            f"Function '{func.__name__}' first parameter is '{params[0]}', "
            "but must be 'image' (3D array: C, Y, X)."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        function_names=[func.__name__],
    )


def validate_function_attributes(func: Callable) -> ValidationResult:
    """
    Validate that function has required memory type attributes.

    Args:
        func: Function to validate

    Returns:
        ValidationResult with attribute validation results
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        metadata = CallableMetadata.from_callable(func)
    except (TypeError, ValueError) as exc:
        errors.append(
            f"Function '{func.__name__}' has an invalid memory declaration: {exc}"
        )
    else:
        required_boundaries = (
            MemoryContractAttribute.INPUT,
            MemoryContractAttribute.OUTPUT,
        )
        missing_boundaries = tuple(
            boundary.value
            for boundary in required_boundaries
            if boundary.read(metadata) is None
        )
        if missing_boundaries:
            errors.append(
                f"Function '{func.__name__}' lacks required memory declarations "
                f"{missing_boundaries!r}. Use an OpenHCS memory decorator such as "
                "@numpy or @cupy."
            )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        function_names=[func.__name__],
    )


def validate_code(code: str) -> ValidationResult:
    """
    Run all code validations before exec().

    Performs syntax, import, and decorator validation in sequence.
    Returns at first validation failure.

    Args:
        code: Python code string to validate

    Returns:
        ValidationResult with combined validation results
    """
    # Validate syntax first
    result = validate_syntax(code)
    if not result.is_valid:
        return result

    # Validate imports
    result = validate_imports(code)
    if not result.is_valid:
        return result

    # Validate decorators
    result = validate_decorator(code)
    if not result.is_valid:
        return result

    return result


def validate_function(func: Callable) -> ValidationResult:
    """
    Validate a function object after exec().

    Checks signature and required attributes.

    Args:
        func: Function to validate

    Returns:
        ValidationResult with combined validation results
    """
    # Validate signature
    result = validate_function_signature(func)
    if not result.is_valid:
        return result

    # Validate attributes
    result = validate_function_attributes(func)
    if not result.is_valid:
        return result

    return result
