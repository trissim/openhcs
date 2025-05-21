"""
Processing Context for OpenHCS.

This module defines the ProcessingContext class, which maintains state during pipeline execution.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext is the canonical owner of all state during pipeline execution.
    Steps should use only context attributes and must not modify context fields
    except through the StepResult mechanism.

    OWNERSHIP: This class may ONLY be instantiated by PipelineOrchestrator.
    All other components must receive a context instance, never create one.

    Attributes:
        step_plans: Dictionary mapping step IDs to execution plans
        outputs: Dictionary for step outputs
        intermediates: Dictionary for intermediate results
        current_step: Current executing step ID
        well_id: Identifier of the well being processed
    """

    def __init__(
        self,
        step_plans: Optional[Dict[str, Dict[str, Any]]] = None,
        well_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the processing context.

        Args:
            step_plans: Dictionary mapping step IDs to execution plans
            well_id: Identifier of the well being processed
            **kwargs: Additional context attributes
        """
        self.step_plans = step_plans or {}
        self.outputs = {}
        self.intermediates = {}
        self.current_step = None
        self.well_id = well_id

        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def inject_plan(self, step_id: str, plan: Dict[str, Any]) -> None:
        """
        Inject a step plan into the context.

        This method is the canonical way to add step plans to the context.
        All step configuration must be injected into the context using this method.

        Args:
            step_id: The unique identifier of the step
            plan: The step execution plan

        # Clause 281 — ProcessingContext is canonical plan authority
        # Clause 504 — Compiler/runtime separation
        # Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        self.step_plans[step_id] = plan

    def update_from_step_result(self, step_result: 'StepResult') -> None:
        """
        Update context from a step result.

        This method applies the context updates from a StepResult object.
        It is the canonical way to update the context from step results.

        Args:
            step_result: StepResult object containing context updates

        # Clause 88 — No Inferred Capabilities
        # Clause 281 — Context-Bound Identifiers
        # Clause 524 — Step = Declaration = ID = Runtime Authority
        """
        # Handle output_path if present
        if step_result.output_path is not None:
            self.outputs['output_path'] = step_result.output_path

        # Update outputs
        for key, value in step_result.results.items():
            self.outputs[key] = value

        # Apply context updates
        for key, value in step_result.context_updates.items():
            setattr(self, key, value)
