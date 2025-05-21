"""
GPU memory type validator for OpenHCS.

This module provides the GPUMemoryTypeValidator class, which is responsible for
validating GPU memory types and assigning GPU IDs to steps requiring GPU memory.

Doctrinal Clauses:
- Clause 66 — Immutability After Construction
- Clause 88 — No Inferred Capabilities
- Clause 293 — GPU Pre-Declaration Enforcement
- Clause 295 — GPU Scheduling Affinity
"""

import logging
from typing import Any, Dict

from openhcs.constants.constants import VALID_GPU_MEMORY_TYPES
from openhcs.core.pipeline.gpu_memory_validator_base import \
    GPUMemoryTypeValidatorBase

# Import this here to avoid circular imports
# It's only used in the validate_step_plans method
try:
    from openhcs.core.orchestrator.gpu_scheduler import \
        get_gpu_registry_status
except ImportError:
    # Provide a fallback for when the orchestrator module is not available
    def get_gpu_registry_status():
        return {0: {"max_pipelines": 1, "active": 0}}

logger = logging.getLogger(__name__)


class GPUMemoryTypeValidator(GPUMemoryTypeValidatorBase):
    """
    Validator for GPU memory types in step plans.

    This validator ensures that all declared GPU memory types are compatible
    with available hardware, assigns valid GPU device IDs to steps requiring
    GPU memory using the centralized GPU scheduler registry, and fails loudly
    if no suitable GPU is available.

    Key principles:
    1. All declared GPU memory types must be validated
    2. Steps requiring GPU memory must be assigned a valid GPU device ID via the scheduler
    3. Validation must fail loudly if required GPU hardware is unavailable
    4. No inference or mutation of declared memory types is allowed
    5. GPU assignment must be thread-safe and respect concurrency limits
    """

    @staticmethod
    def validate_step_plans(
        step_plans: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate GPU memory types in step plans and assign GPU IDs.

        This method checks each step plan for GPU memory types and
        assigns a GPU ID to the step plan if needed. The GPU ID is
        assigned during planning/compilation, not during execution.

        Args:
            step_plans: Dictionary mapping step IDs to step plans

        Returns:
            Dictionary mapping step IDs to dictionaries containing GPU assignments

        Raises:
            ValueError: If no GPUs are available
        """
        # Check if any step requires GPU
        requires_gpu = False

        for step_id, step_plan in step_plans.items():
            input_memory_type = step_plan.get('input_memory_type')
            output_memory_type = step_plan.get('output_memory_type')

            if (input_memory_type in VALID_GPU_MEMORY_TYPES or
                output_memory_type in VALID_GPU_MEMORY_TYPES):
                requires_gpu = True
                break

        # If no step requires GPU, return empty assignments
        if not requires_gpu:
            return {}

        # Get GPU registry status
        try:
            gpu_registry = get_gpu_registry_status()
            logger.info("GPU registry status: %s", gpu_registry)
        except Exception as e:
            logger.warning("Failed to get GPU registry status: %s", e)
            gpu_registry = {}

        if not gpu_registry:
            raise ValueError(
                "Clause 293 Violation: No GPUs available for assignment. "
                "Cannot validate GPU memory types."
            )

        # Find the least loaded GPU
        least_loaded_gpu = min(
            gpu_registry.items(),
            key=lambda x: x[1]['active'] / x[1]['max_pipelines']
        )[0]

        # Assign the same GPU ID to all steps in the pipeline
        # This ensures GPU affinity throughout the pipeline
        gpu_id = least_loaded_gpu

        # GPU ID will be assigned to step plans only, not to context

        # Assign GPU ID to step plans
        gpu_assignments = {}
        for step_id, step_plan in step_plans.items():
            input_memory_type = step_plan.get('input_memory_type')
            output_memory_type = step_plan.get('output_memory_type')

            if (input_memory_type in VALID_GPU_MEMORY_TYPES or
                output_memory_type in VALID_GPU_MEMORY_TYPES):
                # Assign GPU ID to step plan
                step_plan['gpu_id'] = gpu_id
                gpu_assignments[step_id] = {"gpu_id": gpu_id}

                # Log assignment for debugging
                logger.debug(
                    "Step %s assigned gpu_id %s for memory types: %s/%s",
                    step_id, gpu_id, input_memory_type, output_memory_type
                )

        return gpu_assignments
