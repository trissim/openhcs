from typing import List, Dict
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
# Import directly from modules to avoid circular dependency
from openhcs.core.pipeline.compiler import PipelineCompiler
# Removed import of GPUMemoryTypeValidator to break circular dependency
from openhcs.core.pipeline.step_attribute_stripper import \
    StepAttributeStripper


# Define Pipeline class
class Pipeline(list):
    """
    A Pipeline that behaves like List[AbstractStep] but carries metadata.

    This class inherits from list, making it fully compatible with any code
    expecting List[AbstractStep], while providing additional pipeline-specific
    functionality like naming, metadata, and serialization.

    Key Benefits:
    - Drop-in replacement for List[AbstractStep]
    - Backward compatible with existing .steps access
    - Rich metadata support for debugging and UI
    - Method chaining for fluent pipeline construction

    Doctrinal Clauses:
    - Clause 66 — Immutability After Construction (metadata can be set but steps are managed via list methods)
    - Clause 88 — No Inferred Capabilities (explicit type declarations)
    """

    def __init__(self, steps=None, *, name=None, metadata=None, description=None):
        """
        Initialize a pipeline that behaves like a list of steps.

        Args:
            steps: Initial list of AbstractStep objects
            name: Human-readable name for the pipeline
            metadata: Additional metadata dictionary
            description: Optional description of what this pipeline does
        """
        # Initialize the list part with steps
        super().__init__(steps or [])

        # Pipeline metadata
        self.name = name or f"Pipeline_{id(self)}"
        self.description = description
        self.metadata = metadata or {}

        # Add creation timestamp for debugging
        from datetime import datetime
        self.metadata.setdefault('created_at', datetime.now().isoformat())

    @property
    def steps(self):
        """
        Backward compatibility property.

        Returns self since Pipeline IS a list of steps.
        This ensures existing code using pipeline.steps continues to work.
        """
        return self

    def add_step(self, step):
        """
        Add a step to the pipeline and return self for method chaining.

        Args:
            step: AbstractStep to add to the pipeline

        Returns:
            self for fluent method chaining
        """
        self.append(step)
        return self

    def clone(self, *, name=None, metadata=None):
        """
        Create a copy of this pipeline with optional new metadata.

        Args:
            name: New name for the cloned pipeline
            metadata: New metadata (merged with existing)

        Returns:
            New Pipeline instance with copied steps
        """
        new_metadata = self.metadata.copy()
        if metadata:
            new_metadata.update(metadata)

        return Pipeline(
            steps=self.copy(),  # Shallow copy of the step list
            name=name or f"{self.name}_copy",
            metadata=new_metadata,
            description=self.description
        )

    def to_dict(self):
        """
        Convert the pipeline to a dictionary for serialization.

        Returns:
            Dictionary representation of the pipeline
        """
        return {
            "name": self.name,
            "description": self.description,
            "steps": list(self),  # Convert to plain list for serialization
            "metadata": self.metadata,
            "step_count": len(self)
        }

    def __repr__(self):
        """Enhanced string representation for debugging."""
        return f"Pipeline(name='{self.name}', steps={len(self)})"

    def __str__(self):
        """Human-readable string representation."""
        step_summary = f"{len(self)} step{'s' if len(self) != 1 else ''}"
        return f"{self.name} ({step_summary})"

__all__ = [
    # Constants from backends
    'Backend',
    'DEFAULT_BACKEND',
    'REQUIRES_DISK_READ',
    'REQUIRES_DISK_WRITE',
    'FORCE_DISK_WRITE',
    'READ_BACKEND',
    'WRITE_BACKEND',

    # Constants from memory
    'MemoryType',
    'VALID_MEMORY_TYPES',
    'VALID_GPU_MEMORY_TYPES',

    # Core components
    'Pipeline',
    'PipelineCompiler',

    # Planner components
    'PipelinePathPlanner',
    'MaterializationFlagPlanner',
    'FuncStepContractValidator',
    # Removed GPUMemoryTypeValidator to break circular dependency
    'StepAttributeStripper'
]
