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

from openhcs.constants.constants import (FORCE_DISK_WRITE, READ_BACKEND, # DEFAULT_BACKEND removed
                                             REQUIRES_DISK_READ, REQUIRES_DISK_WRITE, WRITE_BACKEND)
from openhcs.core.context.processing_context import ProcessingContext # ADDED
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep
from openhcs.microscopes.microscope_base import MicroscopeHandler

logger = logging.getLogger(__name__)


def _select_first_step_read_backend(context: ProcessingContext, step_index: int) -> str:
    """
    Select the optimal read backend for the first step based on microscope handler compatibility.

    Logic:
    1. If single compatible backend → use it directly
    2. If multiple compatible backends → check metadata availability and select first available
    3. Fail loud if no compatible backends are available

    Args:
        context: ProcessingContext containing microscope handler and input directory
        step_index: Index of the current step (0-based)

    Returns:
        Backend name as string (e.g., "disk", "zarr")

    Raises:
        RuntimeError: If no compatible backends are available or context is invalid
    """
    if step_index != 0:
        return "disk"  # Only first step gets special treatment

    # Validate context has required attributes
    if not hasattr(context, 'microscope_handler') or context.microscope_handler is None:
        error_msg = "ProcessingContext missing microscope_handler - cannot select backend"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not hasattr(context, 'input_dir') or context.input_dir is None:
        error_msg = "ProcessingContext missing input_dir - cannot select backend"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    microscope_handler = context.microscope_handler

    # Validate microscope handler has compatible_backends property
    if not hasattr(microscope_handler, 'compatible_backends'):
        error_msg = f"MicroscopeHandler {type(microscope_handler).__name__} missing compatible_backends property"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    compatible_backends = microscope_handler.compatible_backends

    # Validate compatible_backends is not empty
    if not compatible_backends:
        error_msg = f"MicroscopeHandler {type(microscope_handler).__name__} has empty compatible_backends list"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Single backend: use it directly (ImageXpress, Opera Phenix)
    if len(compatible_backends) == 1:
        selected = compatible_backends[0].value
        logger.debug(f"Single compatible backend: {selected}")
        return selected

    # Multiple backends: check metadata availability (OpenHCS)
    try:
        available_backends = microscope_handler.metadata_handler.get_available_backends(context.input_dir)
        logger.debug(f"Available backends from metadata: {available_backends}")

        # Find first compatible backend that's available
        for backend in compatible_backends:
            if available_backends.get(backend.value, False):
                selected = backend.value
                logger.debug(f"Selected available compatible backend: {selected}")
                return selected

        # No compatible backends available - FAIL LOUD
        available_list = [k for k, v in available_backends.items() if v]
        compatible_list = [b.value for b in compatible_backends]
        error_msg = f"No compatible backends available in metadata. Compatible: {compatible_list}, Available: {available_list}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    except AttributeError as e:
        # Handler doesn't have get_available_backends method (non-OpenHCS)
        # Fall back to first compatible backend
        selected = compatible_backends[0].value
        logger.debug(f"Metadata method not available, using first compatible: {selected}")
        return selected



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
        context: ProcessingContext, # CHANGED: context is now the primary input
        pipeline_definition: List[AbstractStep] # Renamed 'steps' for clarity
        # well_id and step_plans are now derived from context
    ) -> None:
        """
        Prepare materialization flags for each step in a pipeline and injects them
        into context.step_plans.

        This method determines materialization flags and backend selection for each step
        in a pipeline, taking into account step type, position, and declared flags.
        The flags are then added to the corresponding step_plan in context.step_plans.

        Args:
            context: The ProcessingContext, containing step_plans, well_id, and config.
            pipeline_definition: List of AbstractStep instances defining the pipeline.
        """
        step_plans = context.step_plans
        well_id = context.well_id # Used for logging/completeness if step_plan doesn't have it
        vfs_config = context.get_vfs_config()

        if not step_plans:
            logger.warning("Context step_plans is empty. Materialization flags will not be set.")
            return

        # Process each step in the pipeline
        for i, step in enumerate(pipeline_definition): # Use pipeline_definition
            # Get step UID
            step_id = step.step_id
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
            if i == len(pipeline_definition) - 1: # Use pipeline_definition
                requires_disk_output = True
                logger.debug(f"Last step {step_name} always requires disk output")

            # Force disk output if explicitly requested
            if force_disk_output:
                requires_disk_output = True
                logger.debug(f"Step {step_name} has force_disk_output=True, setting requires_disk_output=True")

            # Determine backend selection based on materialization flags
            # Only FunctionStep can use non-disk (intermediate) backends.
            # Default to persistent for safety, then adjust.

            # READ BACKEND determination
            # Check if read_backend was already set by path planner (e.g., chain breaker logic)
            if READ_BACKEND in current_step_plan:
                logger.debug(f"Step {step_name} read_backend already set to '{current_step_plan[READ_BACKEND]}' by path planner, preserving it.")
            elif requires_disk_input: # Includes first step.
                if i == 0:  # First step gets intelligent backend selection
                    selected_backend = _select_first_step_read_backend(context, i)
                    current_step_plan[READ_BACKEND] = selected_backend
                    logger.debug(f"First step {step_name} selected read backend: '{selected_backend}'")
                else:  # Other steps requiring disk input
                    current_step_plan[READ_BACKEND] = "disk"
                    logger.debug(f"Step {step_name} requires disk input, using 'disk' for reading.")
            elif is_function_step: # Can read from an intermediate backend.
                current_step_plan[READ_BACKEND] = vfs_config.default_intermediate_backend
                logger.debug(f"Step {step_name} is FunctionStep and does not require disk input, using '{vfs_config.default_intermediate_backend}' for reading (from default_intermediate_backend).")
            else: # Non-FunctionStep not requiring disk input (e.g., CompositeStep). Must read from persistent store.
                current_step_plan[READ_BACKEND] = "disk" # Default to 'disk' if not intermediate.
                logger.debug(f"Step {step_name} is not FunctionStep and does not require disk input, defaulting to 'disk' for reading.")

            # WRITE BACKEND determination
            if requires_disk_output: # This includes last step, force_disk_output, or step's own requirement.
                current_step_plan[WRITE_BACKEND] = vfs_config.default_materialization_backend
                logger.debug(f"Step {step_name} requires disk output, using '{vfs_config.default_materialization_backend}' for writing (from default_materialization_backend).")
            elif is_function_step: # Not requires_disk_output and is_function_step, so can use intermediate.
                current_step_plan[WRITE_BACKEND] = vfs_config.default_intermediate_backend
                logger.debug(f"Step {step_name} is FunctionStep and does not require disk output, using '{vfs_config.default_intermediate_backend}' for writing (from default_intermediate_backend).")
            else: # Non-FunctionStep not requiring disk output. If it writes primary data, it must be to a persistent store.
                current_step_plan[WRITE_BACKEND] = vfs_config.default_materialization_backend
                logger.debug(f"Step {step_name} is not FunctionStep and does not require disk output, defaulting to '{vfs_config.default_materialization_backend}' for writing (from default_materialization_backend).")

            # Store other flags directly into the step_plan for this step
            current_step_plan[REQUIRES_DISK_READ] = requires_disk_input
            current_step_plan[REQUIRES_DISK_WRITE] = requires_disk_output
            current_step_plan[FORCE_DISK_WRITE] = force_disk_output
            # READ_BACKEND and WRITE_BACKEND are set above

            # Log backend selection
            logger.debug(
                f"Step {step_name} backend selection: "
                f"read_backend={current_step_plan[READ_BACKEND]}, "
                f"write_backend={current_step_plan[WRITE_BACKEND]}"
            )

        # No return value as step_plans is modified in place
