"""
Abstract Step Interface

This module defines the AbstractStep interface, which is the base class for all steps
in the OpenHCS pipeline. It provides the core functionality for step execution,
validation, and state management.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 12 — Absolute Clean Execution
- Clause 21 — Context Immunity
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 92 — Structural Validation First
- Clause 106-A — Declared Memory Types
- Clause 244 — Rot Intolerance
- Clause 245 — Declarative Enforcement
- Clause 246 — Statelessness Mandate
- Clause 251 — Declarative Memory Conversion
- Clause 503 — Cognitive Load Transfer
"""

import abc
import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from openhcs.constants.constants import VariableComponents, GroupBy

# ProcessingContext is used in type hints
if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext
# StepResult is no longer returned by process()


def get_step_id(step: 'AbstractStep') -> str:
    """
    Generate a stable step ID from a step object reference.

    This function provides a deterministic way to derive a step's ID
    from its object reference, enabling stateless execution where
    step objects don't need to store their own IDs as attributes.

    Args:
        step: The step object to generate an ID for

    Returns:
        A stable string ID based on the step object's identity

    Note:
        This uses the same algorithm as step.__init__() to ensure
        consistency between compilation and execution phases.
    """
    return str(id(step))


class AbstractStep(abc.ABC):
    """
    Abstract base class for all steps in the OpenHCS pipeline.

    This class defines the interface that all steps must implement.
    Steps are stateful during pipeline definition and compilation (holding attributes
    like name, input/output memory types, etc.). After compilation, these attributes
    are stripped by the StepAttributeStripper, and the step instances become
    stateless shells. During execution, steps operate solely based on the
    ProcessingContext (which is frozen) and their specific plan within
    context.step_plans.

    # Clause 3 — Declarative Primacy
    # Clause 66 — Immutability After Construction
    # Clause 88 — No Inferred Capabilities
    # Clause 106-A — Declared Memory Types
    # Clause 246 — Statelessness Mandate
    # Clause 251 — Declarative Memory Conversion
    # Clause 503 — Cognitive Load Transfer
    """

    @property
    @abstractmethod
    def requires_disk_input(self) -> bool:
        """Indicates if the step requires its primary input to be on disk."""
        pass

    @property
    @abstractmethod
    def requires_disk_output(self) -> bool:
        """Indicates if the step requires its primary output to be written to disk."""
        pass

    # Step metadata - these are primarily used during pipeline definition and compilation
    step_id: str
    enabled: bool = True
    description: Optional[str] = None
    name: str # Made non-optional, defaults to class name

    # Attributes like input_memory_type, output_memory_type, etc.,
    # are defined in concrete subclasses (e.g., FunctionStep) as needed.

    def __init__(
        self,
        *,  # Force keyword-only arguments
        name: Optional[str] = None,
        variable_components: Optional[List[VariableComponents]] = None,
        force_disk_output: Optional[bool] = False,
        group_by: Optional[GroupBy] = None,
        input_dir: Optional[Union[str,Path]] = None, # Used during path planning
        output_dir: Optional[Union[str,Path]] = None # Used during path planning
    ) -> None:
        """
        Initialize a step. These attributes are primarily used during the
        pipeline definition and compilation phase. After compilation, step
        instances are stripped of these attributes by StepAttributeStripper
        to enforce statelessness during execution.

        Args:
            name: Human-readable name for the step. Defaults to class name.
            variable_components: List of variable components for this step.
            force_disk_output: Whether to force filesystem output.
            group_by: Optional grouping hint for step execution.
            input_dir: Hint for input directory, used by path planner.
            output_dir: Hint for output directory, used by path planner.
        """
        self.name = name or self.__class__.__name__
        self.variable_components = variable_components
        self.force_disk_output = force_disk_output
        self.group_by = group_by
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Generate a stable step_id based on object id at instantiation.
        # This ID is used to link the step object to its plan in the context.
        self.step_id = str(id(self))

        logger_instance = logging.getLogger(__name__)
        logger_instance.debug(f"Created step '{self.name}' (type: {self.__class__.__name__}) with ID {self.step_id}")

    @abc.abstractmethod
    def process(self, context: 'ProcessingContext') -> None:
        """
        Process the step with the given context.

        This method must be implemented by all step subclasses.
        During execution, the step instance is stateless. All necessary
        configuration and paths are retrieved from `context.step_plans[self.step_id]`.
        The `context` itself is frozen and must not be modified.
        Outputs are written to VFS via `context.filemanager` based on the step's plan.
        This method returns None.

        Args:
            context: The frozen ProcessingContext containing all required fields,
                     including step_plans and filemanager.

        # Clause 246 — Statelessness Mandate
        # Clause 21 — Context Immunity (Context is read-only for steps)
        """
        raise NotImplementedError("AbstractStep.process() must be implemented by subclasses")
