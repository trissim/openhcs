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

# Forward reference for StepResult to avoid circular imports
if TYPE_CHECKING:
    from openhcs.core.steps.step_result import StepResult


class AbstractStep(abc.ABC):
    """
    Abstract base class for all steps in the OpenHCS pipeline.

    This class defines the interface that all steps must implement.
    It is a non-instantiable protocol-like enforcement class with:
    - No schema
    - No validator
    - No registration

    Only FunctionStep, PositionGenerationStep, and ImageAssemblyStep receive
    schemas and registration.

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
        pass

    @property
    @abstractmethod
    def requires_disk_output(self) -> bool:
        pass

    # Step metadata
    step_id: str
    enabled: bool = True
    description: Optional[str] = None

    # No capabilities - capabilities are backend concerns, not step concerns

    def __init__(
        self,
        *,  # Force keyword-only arguments
        name: Optional[str] = None,
        variable_components: Optional[List[str]] = None,
        force_disk_output: Optional[bool] = False,  # Required field, no default
        group_by: Optional[str] = None,
        input_dir: Optional[Union[str,Path]] = None,
        output_dir: Optional[Union[str,Path]] = None
    ) -> None:
        """
        Initialize a step with explicit memory type declarations.

        Args:
            name: Human-readable name for the step
            variable_components: List of variable components for this step
            force_disk_output: Whether to force filesystem output regardless of requirements
                (required)
            group_by: Optional grouping hint for step execution (e.g., "channel")
            input_dir: The input directory for the step (required)
            output_dir: The output directory for the step (required)

        Raises:
            TypeError: If input_dir or output_dir are not Path or str
        """

        self.name = name or self.__class__.__name__
        self.variable_components = variable_components
        self.force_disk_output = force_disk_output
        self.group_by = group_by
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Generate a stable step_id based on object id
        self.step_id = str(id(self))

        # Log the step_id for debugging
        logger = logging.getLogger(__name__)
        logger.debug(f"Created step {self.__class__.__name__} with ID {self.step_id}")

    @abc.abstractmethod
    def process(self, context: 'ProcessingContext') -> 'StepResult':
        """
        Process the step with the given context.

        This method must be implemented by all step subclasses.
        It must be stateless and must not modify the context directly.
        All modifications must be returned as part of the StepResult.

        Implementation must validate memory types according to:
        - Clause 106-A — Declared Memory Types
        - Clause 251 — Declarative Memory Conversion
        - Clause 88 — No Inferred Capabilities
        - Clause 503 — Cognitive Load Transfer
        - Clause 65 — Fail Loudly

        Args:
            context: The processing context containing all required fields.

        Returns:
            A StepResult object containing the result of the step execution.

        # Clause 246 — Statelessness Mandate
        # Clause 281 — Context-Bound Identifiers
        # Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        raise NotImplementedError("AbstractStep.process() must be implemented by subclasses")
