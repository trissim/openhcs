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
from pathlib import Path
from typing import Any, Dict, List

from openhcs.constants.constants import READ_BACKEND, WRITE_BACKEND, Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.config import MaterializationBackend

logger = logging.getLogger(__name__)


class MaterializationFlagPlanner:
    """Sets read/write backends for pipeline steps."""

    @staticmethod
    def prepare_pipeline_flags(
        context: ProcessingContext,
        pipeline_definition: List[AbstractStep],
        plate_path: Path
    ) -> None:
        """Set read/write backends for pipeline steps."""

        # === SETUP ===
        vfs_config = context.get_vfs_config()
        step_plans = context.step_plans

        # === PROCESS EACH STEP ===
        for i, step in enumerate(pipeline_definition):
            step_plan = step_plans[i]  # Use step index instead of step_id

            # === READ BACKEND SELECTION ===
            if i == 0:  # First step - read from plate format
                read_backend = MaterializationFlagPlanner._get_first_step_read_backend(context)
                step_plan[READ_BACKEND] = read_backend

                # Zarr conversion flag is already set by path planner if needed
            else:  # Other steps - read from memory (unless already set by chainbreaker logic)
                if READ_BACKEND not in step_plan:
                    step_plan[READ_BACKEND] = Backend.MEMORY.value

            # === WRITE BACKEND SELECTION ===
            # Check if this step will use zarr (has zarr_config set by compiler)
            will_use_zarr = step_plan.get("zarr_config") is not None

            if will_use_zarr:
                # Steps with zarr_config should write to materialization backend
                step_plan[WRITE_BACKEND] = vfs_config.materialization_backend.value
            elif i == len(pipeline_definition) - 1:  # Last step without zarr - write to materialization backend
                step_plan[WRITE_BACKEND] = vfs_config.materialization_backend.value
            else:  # Other steps - write to memory
                step_plan[WRITE_BACKEND] = Backend.MEMORY.value

            # === PER-STEP MATERIALIZATION BACKEND SELECTION ===
            if "materialized_output_dir" in step_plan:
                step_plan["materialized_backend"] = vfs_config.materialization_backend.value

    @staticmethod
    def _get_first_step_read_backend(context: ProcessingContext) -> str:
        """Get read backend for first step based on compatible backends (in priority order) and availability."""
        compatible_backends = context.microscope_handler.compatible_backends

        if len(compatible_backends) == 1:
            # Only one compatible - use its string value
            return compatible_backends[0].value
        else:
            # Multiple compatible - check availability in priority order
            available_backends = context.microscope_handler.metadata_handler.get_available_backends(context.input_dir)

            # Use first compatible backend (highest priority) that's actually available
            for backend_enum in compatible_backends:
                backend_name = backend_enum.value
                if available_backends.get(backend_name, False):
                    return backend_name

            # No compatible backends are available - fail loud
            compatible_names = [b.value for b in compatible_backends]
            raise RuntimeError(f"No compatible backends are actually available. Compatible: {compatible_names}, Available: {available_backends}")






