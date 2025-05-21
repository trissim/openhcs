"""
Pipeline module for the OpenHCS pipeline architecture.

This module provides components for building and executing pipelines,
including compilation, execution, and result handling.

Doctrinal Clauses:
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 17-B — Path Format Discipline
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 106-A — Declared Memory Types
- Clause 251 — Declarative Memory Conversion
- Clause 262 — Pre-Runtime Conversion
- Clause 281 — Context-Bound Identifiers
- Clause 297 — Immutable Result Enforcement
- Clause 504 — Pipeline Preparation Modifications
- Clause 524 — Step = Declaration = ID = Runtime Authority
"""

# Import from constants
from openhcs.constants.constants import (DEFAULT_BACKEND,
                                            ENFORCED_BACKEND_MAPPING,
                                            FORCE_DISK_WRITE, READ_BACKEND,
                                            REQUIRES_DISK_READ,
                                            REQUIRES_DISK_WRITE,
                                            VALID_GPU_MEMORY_TYPES,
                                            VALID_MEMORY_TYPES, WRITE_BACKEND,
                                            Backend, MemoryType)
from openhcs.core.pipeline.funcstep_contract_validator import \
    FuncStepContractValidator
from openhcs.core.pipeline.materialization_flag_planner import \
    MaterializationFlagPlanner
# Import from existing modules
from openhcs.core.pipeline.path_planner import PipelinePathPlanner
# Import from pipeline module
from openhcs.core.pipeline.pipeline import (PipelineCompiler,
                                               PipelineExecutor, StepResult)
# Removed import of GPUMemoryTypeValidator to break circular dependency
from openhcs.core.pipeline.step_attribute_stripper import \
    StepAttributeStripper


# Define Pipeline class
class Pipeline:
    """
    Pipeline class for OpenHCS.

    This class represents a pipeline with steps and metadata.

    Attributes:
        name: Name of the pipeline
        steps: List of steps in the pipeline
        metadata: Additional metadata for the pipeline
    """

    def __init__(self, name=None, steps=None, metadata=None):
        """
        Initialize a pipeline.

        Args:
            name: Name of the pipeline
            steps: List of steps in the pipeline
            metadata: Additional metadata for the pipeline
        """
        self.name = name or "Unnamed Pipeline"
        self.steps = steps or []
        self.metadata = metadata or {}

    def add_step(self, step):
        """
        Add a step to the pipeline.

        Args:
            step: Step to add to the pipeline

        Returns:
            self for method chaining
        """
        self.steps.append(step)
        return self

    def to_dict(self):
        """
        Convert the pipeline to a dictionary.

        Returns:
            Dictionary representation of the pipeline
        """
        return {
            "name": self.name,
            "steps": self.steps,
            "metadata": self.metadata
        }

__all__ = [
    # Constants from backends
    'Backend',
    'DEFAULT_BACKEND',
    'REQUIRES_DISK_READ',
    'REQUIRES_DISK_WRITE',
    'FORCE_DISK_WRITE',
    'READ_BACKEND',
    'WRITE_BACKEND',
    'ENFORCED_BACKEND_MAPPING',

    # Constants from memory
    'MemoryType',
    'VALID_MEMORY_TYPES',
    'VALID_GPU_MEMORY_TYPES',

    # Core components
    'Pipeline',
    'StepResult',
    'PipelineCompiler',
    'PipelineExecutor',

    # Planner components
    'PipelinePathPlanner',
    'MaterializationFlagPlanner',
    'FuncStepContractValidator',
    # Removed GPUMemoryTypeValidator to break circular dependency
    'StepAttributeStripper'
]
