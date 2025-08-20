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
from openhcs.core.utils import optional_import

# LAZY IMPORT: Import gpu_scheduler only when needed to avoid circular dependency
# from openhcs.core.orchestrator.gpu_scheduler import get_gpu_registry_status

logger = logging.getLogger(__name__)


def _validate_required_libraries(required_libraries: set) -> None:
    """
    Validate that required GPU libraries are installed.

    Args:
        required_libraries: Set of memory types that require library validation

    Raises:
        ValueError: If any required library is not installed
    """
    missing_libraries = []

    for memory_type in required_libraries:
        if memory_type == "cupy":
            cupy = optional_import("cupy")
            if cupy is None:
                missing_libraries.append("cupy")
        elif memory_type == "torch":
            torch = optional_import("torch")
            if torch is None:
                missing_libraries.append("torch")
        elif memory_type == "tensorflow":
            tensorflow = optional_import("tensorflow")
            if tensorflow is None:
                missing_libraries.append("tensorflow")
        elif memory_type == "jax":
            jax = optional_import("jax")
            if jax is None:
                missing_libraries.append("jax")

    if missing_libraries:
        raise ValueError(
            f"🔥 COMPILATION FAILED: Required GPU libraries not installed: {', '.join(missing_libraries)}. "
            f"Pipeline contains functions decorated with @{'/'.join(missing_libraries)}_func but the corresponding "
            f"libraries are not available. Install the missing libraries or remove the functions from your pipeline."
        )


class GPUMemoryTypeValidator:
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
        step_plans: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Validate GPU memory types in step plans and assign GPU IDs.

        This method checks each step plan for GPU memory types and
        assigns a GPU ID to the step plan if needed. The GPU ID is
        assigned during planning/compilation, not during execution.

        Args:
            step_plans: Dictionary mapping step indices to step plans

        Returns:
            Dictionary mapping step indices to dictionaries containing GPU assignments

        Raises:
            ValueError: If no GPUs are available
        """
        # Check if any step requires GPU and validate library availability
        requires_gpu = False
        required_libraries = set()

        for step_index, step_plan in step_plans.items():
            input_memory_type = step_plan.get('input_memory_type')
            output_memory_type = step_plan.get('output_memory_type')

            if input_memory_type in VALID_GPU_MEMORY_TYPES:
                requires_gpu = True
                required_libraries.add(input_memory_type)

            if output_memory_type in VALID_GPU_MEMORY_TYPES:
                requires_gpu = True
                required_libraries.add(output_memory_type)

        # If no step requires GPU, return empty assignments
        if not requires_gpu:
            return {}

        # Validate that required libraries are installed
        _validate_required_libraries(required_libraries)

        # Get GPU registry status (lazy import to avoid circular dependency)
        try:
            from openhcs.core.orchestrator.gpu_scheduler import get_gpu_registry_status
            gpu_registry = get_gpu_registry_status()
            logger.info("GPU registry status: %s", gpu_registry)
        except Exception as e:
            raise ValueError(f"🔥 COMPILATION FAILED: Cannot access GPU registry: {e}. GPU functions require initialized GPU registry!") from e

        if not gpu_registry:
            raise ValueError(
                "🔥 COMPILATION FAILED: No GPUs available in registry but pipeline contains GPU-decorated functions (@torch, @cupy, etc.)!"
            )

        # Assign the first available GPU (since actual load tracking was orphaned)
        # GPU assignment happens at compilation time, not runtime
        least_loaded_gpu = list(gpu_registry.keys())[0]

        # Assign the same GPU ID to all steps in the pipeline
        # This ensures GPU affinity throughout the pipeline
        gpu_id = least_loaded_gpu

        # GPU ID will be assigned to step plans only, not to context

        # Assign GPU ID to step plans
        gpu_assignments = {}
        for step_index, step_plan in step_plans.items():
            input_memory_type = step_plan.get('input_memory_type')
            output_memory_type = step_plan.get('output_memory_type')

            if (input_memory_type in VALID_GPU_MEMORY_TYPES or
                output_memory_type in VALID_GPU_MEMORY_TYPES):
                # Assign GPU ID to step plan
                step_plan['gpu_id'] = gpu_id
                gpu_assignments[step_index] = {"gpu_id": gpu_id}

                # Log assignment for debugging
                logger.debug(
                    "Step %s assigned gpu_id %s for memory types: %s/%s",
                    step_index, gpu_id, input_memory_type, output_memory_type
                )

        return gpu_assignments
