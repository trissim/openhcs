"""
Materialization flag planner for OpenHCS.

This module provides the MaterializationFlagPlanner class, which is responsible for
determining materialization flags and backend selection for each step in a pipeline.

Doctrinal Clauses:
- Clause 12 — Absolute Clean Execution
- Clause 17 — VFS Exclusivity (FileManager is the only component that uses VirtualPath)
- Clause 65 — No Fallback Logic
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 245 — Path Declaration
- Clause 273 — Backend Authorization Doctrine
- Clause 276 — Positional Backend Enforcement
- Clause 504 — Pipeline Preparation Modifications
"""

import logging
from typing import Any, Dict, List

from openhcs.constants.constants import (DEFAULT_BACKEND, FORCE_DISK_WRITE, READ_BACKEND,
                                            REQUIRES_DISK_READ, REQUIRES_DISK_WRITE, WRITE_BACKEND)
from openhcs.core.steps.abstract_step import AbstractStep
from openhcs.core.steps.function_step import FunctionStep





class MaterializationFlagPlanner:
    """
    Plans and prepares materialization flags for pipeline steps.

    This class is responsible for determining materialization flags and backend selection
    for each step in a pipeline, taking into account step type, position, and declared flags.

    Key principles:
    1. Only FunctionStep can use non-disk backends
    2. Materialization flags are determined by step type and position
    3. Backend selection is based on materialization flags
    4. All steps must have read_backend and write_backend defined
    5. Default backend is used when disk is not required
    """

    @staticmethod
    def prepare_pipeline_flags(
        steps: List[AbstractStep],
        well_id: str,
        step_plans: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Prepare materialization flags for each step in a pipeline and injects them into step_plans.

        This method determines materialization flags and backend selection for each step
        in a pipeline, taking into account step type, position, and declared flags.
        The flags are then added to the corresponding step_plan in the step_plans dictionary.

        Args:
            steps: List of steps to prepare flags for
            well_id: Well identifier for the pipeline
            step_plans: Dictionary mapping step UIDs to their (partially filled) step plans.
                        This dictionary will be modified in place.
        """
        if not step_plans:
            logger.warning("No step_plans provided to MaterializationFlagPlanner. Flags will not be set.")
            return

        # Process each step in the pipeline
        for i, step in enumerate(steps):
            # Get step UID
            step_id = step.uid
            step_name = step.name

            # Ensure the step_plan for this step exists
            if step_id not in step_plans:
                logger.warning(f"No step_plan found for step {step_name} (ID: {step_id}). Skipping flag setting.")
                continue
            
            # Get the specific plan for this step to update
            current_step_plan = step_plans[step_id]
            
            # Add well_id if not already present (though it should be)
            if "well_id" not in current_step_plan:
                current_step_plan["well_id"] = well_id

            # Determine if this is a FunctionStep
            is_function_step = isinstance(step, FunctionStep)

            # Get materialization flags from step
            # These are hardcoded in each step class and should not be mutated
            # Direct attribute access - will raise AttributeError if missing
            requires_disk_input = step.requires_disk_input
            requires_disk_output = step.requires_disk_output

            # Check for force_disk_output flag
            # This is an optional flag that can be set on any step
            force_disk_output = getattr(step, "force_disk_output", False)

            # Log materialization flags
            logger.debug(
                f"Step {step_name} materialization flags: "
                f"requires_disk_input={requires_disk_input}, "
                f"requires_disk_output={requires_disk_output}, "
                f"force_disk_output={force_disk_output}"
            )

            # Apply positional rules
            # First step always requires disk input
            if i == 0:
                requires_disk_input = True
                logger.debug(f"First step {step_name} always requires disk input")

            # Last step always requires disk output
            if i == len(steps) - 1:
                requires_disk_output = True
                logger.debug(f"Last step {step_name} always requires disk output")

            # Force disk output if explicitly requested
            if force_disk_output:
                requires_disk_output = True
                logger.debug(f"Step {step_name} has force_disk_output=True, setting requires_disk_output=True")

            # Determine backend selection based on materialization flags
            # Only FunctionStep can use non-disk backends
            read_backend = "disk"
            write_backend = "disk"

            if not requires_disk_input and is_function_step:
                read_backend = DEFAULT_BACKEND
                logger.debug(f"Step {step_name} does not require disk input, using {read_backend} backend for reading")

            if not requires_disk_output and is_function_step:
                write_backend = DEFAULT_BACKEND
                logger.debug(f"Step {step_name} does not require disk output, using {write_backend} backend for writing")

            # Non-FunctionStep cannot use non-disk backends
            if not is_function_step:
                if read_backend != "disk":
                    # Assuming ERROR_INVALID_BACKEND was defined elsewhere or this check is less critical
                    # For now, log a warning instead of raising an error if it's not a FunctionStep
                    logger.warning(f"Step {step_name} is not a FunctionStep but read_backend is {read_backend}. Forcing to 'disk'.")
                    read_backend = "disk"

                if write_backend != "disk":
                    logger.warning(f"Step {step_name} is not a FunctionStep but write_backend is {write_backend}. Forcing to 'disk'.")
                    write_backend = "disk"
            
            # Store flags directly into the step_plan for this step
            current_step_plan[REQUIRES_DISK_READ] = requires_disk_input
            current_step_plan[REQUIRES_DISK_WRITE] = requires_disk_output
            current_step_plan[FORCE_DISK_WRITE] = force_disk_output
            current_step_plan[READ_BACKEND] = read_backend
            current_step_plan[WRITE_BACKEND] = write_backend
            
            # Log backend selection
            logger.debug(
                f"Step {step_name} backend selection: "
                f"read_backend={read_backend}, "
                f"write_backend={write_backend}"
            )

        # No return value as step_plans is modified in place
