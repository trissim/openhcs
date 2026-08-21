"""
Pipeline module for the OpenHCS pipeline architecture.

This module provides components for building and executing pipelines,
including compilation, execution, and result handling.
"""

from openhcs.constants.constants import (
    DEFAULT_BACKEND,
    FORCE_DISK_WRITE,
    READ_BACKEND,
    REQUIRES_DISK_READ,
    REQUIRES_DISK_WRITE,
    VALID_GPU_MEMORY_TYPES,
    VALID_MEMORY_TYPES,
    WRITE_BACKEND,
    Backend,
    MemoryType,
)
from openhcs.core.pipeline.funcstep_contract_validator import (
    FuncStepContractValidator,
)
from openhcs.core.pipeline.materialization_flag_planner import (
    MaterializationFlagPlanner,
)
# Import from existing modules
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
# Import directly from modules to avoid circular dependency
from openhcs.core.pipeline.compiler import PipelineCompiler
from openhcs.core.pipeline.step_attribute_stripper import StepAttributeStripper

PipelinePublicBinding = tuple[str, object]


def build_all(bindings: tuple[PipelinePublicBinding, ...]) -> list[str]:
    """Build the module export list from the public binding schema."""
    return [name for name, _binding in bindings]


PIPELINE_PUBLIC_BINDINGS: tuple[PipelinePublicBinding, ...] = (
    ("Backend", Backend),
    ("DEFAULT_BACKEND", DEFAULT_BACKEND),
    ("REQUIRES_DISK_READ", REQUIRES_DISK_READ),
    ("REQUIRES_DISK_WRITE", REQUIRES_DISK_WRITE),
    ("FORCE_DISK_WRITE", FORCE_DISK_WRITE),
    ("READ_BACKEND", READ_BACKEND),
    ("WRITE_BACKEND", WRITE_BACKEND),
    ("MemoryType", MemoryType),
    ("VALID_MEMORY_TYPES", VALID_MEMORY_TYPES),
    ("VALID_GPU_MEMORY_TYPES", VALID_GPU_MEMORY_TYPES),
    ("PipelineCompiler", PipelineCompiler),
    ("PipelinePathPlanner", PipelinePathPlanner),
    ("MaterializationFlagPlanner", MaterializationFlagPlanner),
    ("FuncStepContractValidator", FuncStepContractValidator),
    ("StepAttributeStripper", StepAttributeStripper),
)

__all__ = build_all(PIPELINE_PUBLIC_BINDINGS)
