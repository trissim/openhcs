"""
Validation package for openhcs.

This package provides tools for validating openhcs code against architectural
principles using static AST-based analysis.
"""

from openhcs.validation.ast_validator import (
    ValidationViolation,
    validate_file,
    validate_path_types,
    validate_backend_parameter
)

__all__ = [
    'ValidationViolation',
    'validate_file',
    'validate_path_types',
    'validate_backend_parameter'
]
