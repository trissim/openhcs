"""
Pipeline module for the OpenHCS pipeline architecture.

This module provides components for building and executing pipelines,
including compilation, execution, and result handling.
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

PipelinePublicBinding = tuple[str, object]


def build_all(bindings: tuple[PipelinePublicBinding, ...]) -> list[str]:
    """Build the module export list from the public binding schema."""
    return [name for name, _binding in bindings]


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
    """

    def __init__(self, steps=None, *, name=None, metadata=None, description=None, step_scope_ids=None):
        """
        Initialize a pipeline that behaves like a list of steps.

        Args:
            steps: Initial list of AbstractStep objects
            name: Human-readable name for the pipeline
            metadata: Additional metadata dictionary
            description: Optional description of what this pipeline does
            step_scope_ids: List of ObjectState scope IDs for steps (for UI state tracking)
        """
        # Initialize the list part with steps
        super().__init__(steps or [])

        # Pipeline metadata
        self.name = name or f"Pipeline_{id(self)}"
        self.description = description
        self.metadata = metadata or {}

        # ObjectState tracking - list of scope IDs for steps
        self.step_scope_ids = step_scope_ids or []

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
    ("Pipeline", Pipeline),
    ("PipelineCompiler", PipelineCompiler),
    ("PipelinePathPlanner", PipelinePathPlanner),
    ("MaterializationFlagPlanner", MaterializationFlagPlanner),
    ("FuncStepContractValidator", FuncStepContractValidator),
    ("StepAttributeStripper", StepAttributeStripper),
)

__all__ = build_all(PIPELINE_PUBLIC_BINDINGS)
