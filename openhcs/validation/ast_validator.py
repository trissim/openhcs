"""
AST-based validation for openhcs.

This module provides AST-based validation tools for enforcing type safety,
backend parameter validation, and architectural constraints at compile time.
"""

import ast
from dataclasses import dataclass
from enum import Enum
import functools
import os
from typing import ClassVar, Iterable, List, Optional

from metaclass_registry import AutoRegisterMeta


class ValidationKind(str, Enum):
    """Nominal validator family keys used for AST validation."""

    PATH_TYPE = "path_type"
    BACKEND_PARAM = "backend_param"
    MEMORY_TYPE = "memory_type"
    VFS_BOUNDARY = "vfs_boundary"


# Compatibility aliases for existing string comparisons and CLI output.
PATH_TYPE = ValidationKind.PATH_TYPE.value
BACKEND_PARAM = ValidationKind.BACKEND_PARAM.value
MEMORY_TYPE = ValidationKind.MEMORY_TYPE.value
VFS_BOUNDARY = ValidationKind.VFS_BOUNDARY.value

DEFAULT_VALIDATION_KINDS: tuple[ValidationKind, ...] = (
    ValidationKind.PATH_TYPE,
    ValidationKind.BACKEND_PARAM,
    ValidationKind.VFS_BOUNDARY,
    ValidationKind.MEMORY_TYPE,
)

# Error messages
ERROR_INVALID_PATH_TYPE = (
    "Invalid type for {0}: expected str or Path, got {1}. "
    "Only str and Path types are allowed, no automatic conversion is performed."
)
ERROR_MISSING_BACKEND = (
    "Missing required backend parameter. "
    "Backend must be provided as a positional parameter."
)
ERROR_INVALID_BACKEND_TYPE = (
    "Invalid type for backend parameter: expected str, got {0}."
)
ERROR_VFS_BOUNDARY = (
    "VFS Boundary violation: {0}"
)
ERROR_MEMORY_TYPE = (
    "Memory type violation: {0}"
)

@dataclass(frozen=True, slots=True)
class ValidationViolation:
    """Represents a validation violation found during AST analysis."""

    file_path: str
    line_number: int
    violation_type: str | ValidationKind
    message: str
    node: Optional[ast.AST] = None

    def __post_init__(self) -> None:
        if isinstance(self.violation_type, ValidationKind):
            object.__setattr__(self, "violation_type", self.violation_type.value)

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number} - {self.violation_type}: {self.message}"


class ASTValidator(ast.NodeVisitor, metaclass=AutoRegisterMeta):
    """Base AST validator for static code analysis."""

    __registry_key__ = "validation_kind"
    __skip_if_no_key__ = True

    __registry__: ClassVar[dict[ValidationKind, type["ASTValidator"]]] = {}
    validation_kind: ClassVar[ValidationKind | None] = None

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.violations: List[ValidationViolation] = []
        self.current_function: Optional[ast.FunctionDef] = None

    @classmethod
    def validators_for(
        cls,
        kinds: Iterable[ValidationKind | str] | None = None,
    ) -> tuple[type["ASTValidator"], ...]:
        """Return validator classes in stable execution order."""
        selected_kinds = DEFAULT_VALIDATION_KINDS if kinds is None else tuple(
            ValidationKind(kind) for kind in kinds
        )
        return tuple(cls.__registry__[kind] for kind in selected_kinds)

    def add_violation(self, 
                      node: ast.AST, 
                      violation_type: ValidationKind | str,
                      message: str) -> None:
        """Add a validation violation."""
        self.violations.append(
            ValidationViolation(
                file_path=self.file_path,
                line_number=getattr(node, 'lineno', 0),
                violation_type=violation_type,
                message=message,
                node=node
            )
        )
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions to check for decorators and annotations."""
        old_function = self.current_function
        self.current_function = node
        self.generic_visit(node)
        self.current_function = old_function


class PathTypeValidator(ASTValidator):
    """Validates that path parameters are correctly typed as str or Path."""

    validation_kind = ValidationKind.PATH_TYPE
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function parameters for path type annotations."""
        super().visit_FunctionDef(node)
        
        # Check for @validate_path_types decorator
        has_validator = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                if decorator.func.id == 'validate_path_types':
                    has_validator = True
                    break
        
        if not has_validator:
            return
        
        # Check parameter annotations
        for arg in node.args.args:
            if not arg.annotation:
                continue
            
            # Check if parameter is annotated as Union[str, Path]
            if isinstance(arg.annotation, ast.Subscript):
                if isinstance(arg.annotation.value, ast.Name) and arg.annotation.value.id == 'Union':
                    # Check if Union contains str and Path
                    if isinstance(arg.annotation.slice, ast.Index):  # Python < 3.9
                        slice_value = arg.annotation.slice.value
                    else:  # Python >= 3.9
                        slice_value = arg.annotation.slice
                    
                    if isinstance(slice_value, ast.Tuple):
                        types = [elt.id for elt in slice_value.elts if isinstance(elt, ast.Name)]
                        if 'str' in types and 'Path' in types:
                            continue
            
            # Check if parameter is annotated as str or Path
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id in ('str', 'Path'):
                continue
            
            # If we get here, the parameter has an invalid type annotation
            self.add_violation(
                node=arg,
                violation_type=PATH_TYPE,
                message=f"Parameter '{arg.arg}' should be annotated as Union[str, Path], str, or Path"
            )


class BackendParameterValidator(ASTValidator):
    """Validates that backend parameters are correctly passed and typed."""

    validation_kind = ValidationKind.BACKEND_PARAM
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check function calls for backend parameter usage."""
        self.generic_visit(node)
        
        # Check if this is a call to a FileManager method
        if not isinstance(node.func, ast.Attribute):
            return
        
        # Check if the method belongs to FileManager
        if not isinstance(node.func.value, ast.Name):
            return
        
        # List of FileManager methods that require a backend parameter
        filemanager_methods = {
            'list_files', 'list_image_files', 'list_dir', 'ensure_directory',
            'exists', 'rename', 'mirror_directory_with_symlinks', 'create_symlink',
            'delete', 'copy_file', 'open_file', 'save', 'load'
        }
        
        if node.func.attr in filemanager_methods:
            # Check if backend parameter is provided
            if not node.args or len(node.args) < 2:
                self.add_violation(
                    node=node,
                    violation_type=BACKEND_PARAM,
                    message=f"Missing backend parameter in call to '{node.func.attr}'"
                )
                return
            
            # Check if backend parameter is a string literal or a variable
            backend_arg = node.args[-1]  # Backend should be the last positional argument
            if isinstance(backend_arg, ast.Constant) and not isinstance(backend_arg.value, str):
                self.add_violation(
                    node=node,
                    violation_type=BACKEND_PARAM,
                    message=f"Backend parameter must be a string, got {type(backend_arg.value).__name__}"
                )


class VFSBoundaryValidator(ASTValidator):
    """Validates VFS boundary enforcement rules."""

    validation_kind = ValidationKind.VFS_BOUNDARY
    
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.forbidden_imports = {"os.path", "pathlib"}
        self.forbidden_path_constructors = {"pathlib.Path"}
        self.in_test_module = "/tests/" in file_path or os.path.basename(file_path).startswith("test_")
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for forbidden imports."""
        self.generic_visit(node)
        
        if self.in_test_module:
            return
        
        if node.module in self.forbidden_imports:
            self.add_violation(
                node=node,
                violation_type=VFS_BOUNDARY,
                message=f"Forbidden import from '{node.module}'"
            )
    
    def visit_Import(self, node: ast.Import) -> None:
        """Check for forbidden imports."""
        self.generic_visit(node)
        
        if self.in_test_module:
            return
        
        for alias in node.names:
            if alias.name in self.forbidden_imports:
                self.add_violation(
                    node=node,
                    violation_type=VFS_BOUNDARY,
                    message=f"Forbidden import of '{alias.name}'"
                )
    
    def visit_Call(self, node: ast.Call) -> None:
        """Check for forbidden path constructors and to_os_path() calls."""
        self.generic_visit(node)
        
        if self.in_test_module:
            return
        
        # Check for Path() constructor
        if isinstance(node.func, ast.Name) and node.func.id == 'Path':
            self.add_violation(
                node=node,
                violation_type=VFS_BOUNDARY,
                message="Direct Path constructor usage is forbidden"
            )
        
        # Check for to_os_path() method calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to_os_path':
            self.add_violation(
                node=node,
                violation_type=VFS_BOUNDARY,
                message="to_os_path() method can only be used in functions decorated with @vfs_escape_hatch"
            )


class MemoryTypeValidator(ASTValidator):
    """Validates memory type declarations and usage."""

    validation_kind = ValidationKind.MEMORY_TYPE
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check function decorators for memory type declarations."""
        super().visit_FunctionDef(node)
        
        # Check for memory type decorators
        memory_decorators = {'numpy', 'cupy', 'torch', 'tensorflow', 'jax'}
        has_memory_decorator = False
        
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id in memory_decorators:
                has_memory_decorator = True
                break
            elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id in memory_decorators:
                has_memory_decorator = True
                break
        
        # Check if function is in processing module and missing memory decorator
        if not has_memory_decorator and 'processing' in self.file_path:
            # Skip if function is private or a method
            if not node.name.startswith('_') and self.current_function is None:
                self.add_violation(
                    node=node,
                    violation_type=MEMORY_TYPE,
                    message=f"Function '{node.name}' in processing module should have a memory type decorator"
                )


# Decorator functions for runtime validation

def validate_path_types(**type_annotations):
    """
    Decorator to validate path type parameters.
    
    Args:
        **type_annotations: Type annotations for parameters.
    
    Returns:
        Decorated function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Runtime validation can be added here if needed
            return func(*args, **kwargs)
        
        # Store type annotations for AST analysis
        wrapper.__path_type_annotations__ = type_annotations
        return wrapper
    
    return decorator


def validate_backend_parameter(func):
    """
    Decorator to validate backend parameter.
    
    Args:
        func: Function to decorate.
    
    Returns:
        Decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Runtime validation can be added here if needed
        return func(*args, **kwargs)
    
    # Mark function for AST analysis
    wrapper.__validate_backend__ = True
    return wrapper


def run_ast_validators(
    tree: ast.AST,
    file_path: str,
    kinds: Iterable[ValidationKind | str] | None = None,
) -> tuple[ValidationViolation, ...]:
    """Run the registered AST validator family for a parsed source tree."""
    violations: list[ValidationViolation] = []
    for validator_cls in ASTValidator.validators_for(kinds):
        validator = validator_cls(file_path)
        validator.visit(tree)
        violations.extend(validator.violations)
    return tuple(violations)


# Main validation function

def validate_file(
    file_path: str,
    kinds: Iterable[ValidationKind | str] | None = None,
) -> List[ValidationViolation]:
    """
    Validate a Python file using AST-based analysis.
    
    Args:
        file_path: Path to the Python file.
    
    Returns:
        List of validation violations.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        return [ValidationViolation(
            file_path=file_path,
            line_number=e.lineno or 0,
            violation_type="syntax_error",
            message=f"Syntax error: {e}",
            node=None
        )]
    
    return list(run_ast_validators(tree, file_path, kinds))
