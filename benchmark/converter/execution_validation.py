"""Legacy facade for CellProfiler execution validation semantics."""

from openhcs.interop.cellprofiler.execution_validation import (
    CPPipeExecutionValidation,
    CPPipeExecutionValidationError,
    CPPipeInfrastructureProfile,
    validate_cppipe_execution,
    validate_cppipe_runtime_observation,
)

__all__ = [
    "CPPipeExecutionValidation",
    "CPPipeExecutionValidationError",
    "CPPipeInfrastructureProfile",
    "validate_cppipe_execution",
    "validate_cppipe_runtime_observation",
]
