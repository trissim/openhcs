"""
Step attribute stripper for OpenHCS.

This module provides the StepAttributeStripper class, which is responsible for
stripping all attributes from Step instances after planning, ensuring that steps
are attribute-less shells that operate solely via the processing context.

Doctrinal Clauses:
- Clause 12 — Absolute Clean Execution
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Plan Context Supremacy
- Clause 503 — Cognitive Load Transfer
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Error messages for doctrinal violations
ERROR_ATTRIBUTE_DELETION_FAILED = (
    "Clause 66 Violation: Failed to delete attribute '{0}' from step '{1}'. "
    "All attributes must be stripped after planning."
)

ERROR_RESERVED_ATTRIBUTE = (
    "Clause 245 Violation: Step '{0}' has reserved attribute '{1}' that cannot be deleted. "
    "This indicates a design flaw in the step implementation."
)


class StepAttributeStripper:
    """
    Planner that strips all attributes from Step instances after planning.

    This planner ensures that steps are attribute-less shells that operate
    solely via the processing context, with no mutable state or pre-declared
    fields accessible during execution.

    Key principles:
    1. All attributes must be stripped from steps after planning
    2. Steps must operate solely via the processing context
    3. No mutable state or pre-declared fields can be accessed during execution
    4. Execution must be based solely on the plan context
    """

    @staticmethod
    def strip_step_attributes(steps: List[Any], step_plans: Dict[str, Dict[str, Any]]) -> None:
        """
        Strip all attributes from Step instances after planning.

        Args:
            steps: List of Step instances
            step_plans: Dictionary mapping step UIDs to step plans

        Raises:
            ValueError: If attribute deletion fails
            RuntimeError: If a step has reserved attributes that cannot be deleted
        """
        if not steps:
            logger.warning("No steps provided to StepAttributeStripper")
            return

        # Process each step
        for step in steps:
            # Get step identifier for error messages
            step_id = getattr(step, "step_id", str(id(step)))
            step_name = getattr(step, "name", f"Step {step_id}")

            # Get all attributes
            attributes = set(vars(step).keys())

            # Log attributes being stripped
            logger.debug(f"Stripping {len(attributes)} attributes from step '{step_name}': {attributes}")

            # Delete all attributes
            for attr in list(attributes):
                try:
                    delattr(step, attr)
                except (AttributeError, TypeError) as e:
                    # Check if this is a reserved attribute that cannot be deleted
                    if hasattr(type(step), attr) and not hasattr(type(step), "__slots__"):
                        # This is likely a class attribute or method, not an instance attribute
                        logger.debug(f"Skipping class attribute/method '{attr}' on step '{step_name}'")
                        continue

                    # If deletion failed for other reasons, raise an error
                    if hasattr(type(step), "__slots__") and attr in getattr(type(step), "__slots__", []):
                        raise RuntimeError(ERROR_RESERVED_ATTRIBUTE.format(step_name, attr)) from e
                    else:
                        raise ValueError(ERROR_ATTRIBUTE_DELETION_FAILED.format(attr, step_name)) from e

            # Verify that all attributes have been stripped
            remaining_attrs = set(vars(step).keys())
            if remaining_attrs:
                raise ValueError(
                    f"Clause 66 Violation: Step '{step_name}' still has attributes after stripping: {remaining_attrs}. "
                    f"All attributes must be stripped after planning."
                )

            logger.info(f"Successfully stripped all attributes from step '{step_name}'")
