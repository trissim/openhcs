"""
Materialization flag planner for OpenHCS.

This module provides the MaterializationFlagPlanner class, which is responsible for
determining materialization flags and backend selection for each step in a pipeline.
"""

import logging
import dataclasses
from pathlib import Path
from typing import List

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
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
        plate_path: Path,
        pipeline_config
    ) -> None:
        """
        Set read/write backends for pipeline steps.

        Args:
            context: ProcessingContext with step_plans
            pipeline_definition: List of pipeline steps
            plate_path: Path to plate data
            pipeline_config: Merged GlobalPipelineConfig, not the raw
                           PipelineConfig. This ensures proper inheritance.
        """

        # === SETUP ===
        # CRITICAL: pipeline_config is the merged GlobalPipelineConfig.
        # This ensures inheritance without field-specific resolution here.
        vfs_config = pipeline_config.vfs_config
        step_plans = context.step_plans

        last_image_materialization_step = (
            MaterializationFlagPlanner._last_image_materialization_step(
                step_plans,
                len(pipeline_definition),
            )
        )

        # === PROCESS EACH STEP ===
        for i, step in enumerate(pipeline_definition):
            step_plan = step_plans[i]  # Use step index instead of step_id

            # === READ BACKEND SELECTION ===
            if i == 0:  # First step - read from plate format
                read_backend = MaterializationFlagPlanner._get_first_step_read_backend(context, vfs_config)
                step_plan.read_backend = read_backend

                # Zarr conversion flag is already set by path planner if needed
            else:  # Other steps - read from memory (unless already set by chainbreaker logic)
                if step_plan.read_backend is None:
                    # Check if this step reads from PIPELINE_START (original input)
                    from openhcs.core.steps.abstract import InputSource
                    if step.processing_config.input_source == InputSource.PIPELINE_START:
                        # Check if input conversion will happen - if so, use zarr backend
                        if step_plans[0].input_conversion is not None:
                            step_plan.read_backend = Backend.ZARR.value
                            # Also update input_dir to point to conversion target
                            step_plan.input_dir = step_plans[0].input_conversion.output_dir
                            logger.debug(
                                "Step %s: PIPELINE_START with conversion -> zarr backend, input_dir=%s",
                                i,
                                step_plan.input_dir,
                            )
                        else:
                            # No conversion - use the same backend as the first step
                            step_plan.read_backend = step_plans[0].read_backend
                    else:
                        step_plan.read_backend = Backend.MEMORY.value

            # === WRITE BACKEND SELECTION ===
            # Check if this step will use zarr (has zarr_config set by compiler)
            will_use_zarr = step_plan.zarr_config is not None

            if will_use_zarr:
                # Steps with zarr_config should write to materialization backend
                materialization_backend = MaterializationFlagPlanner._resolve_materialization_backend(context, vfs_config)
                step_plan.write_backend = materialization_backend
            elif (
                i == last_image_materialization_step
            ):  # Last image-producing step without zarr - write to materialization backend
                materialization_backend = MaterializationFlagPlanner._resolve_materialization_backend(context, vfs_config)
                step_plan.write_backend = materialization_backend
            else:  # Other steps - write to memory
                step_plan.write_backend = Backend.MEMORY.value

            # === PER-STEP MATERIALIZATION BACKEND SELECTION ===
            if step_plan.materialized_output is not None:
                materialization_backend = MaterializationFlagPlanner._resolve_materialization_backend(context, vfs_config)
                step_plan.materialized_output = dataclasses.replace(
                    step_plan.materialized_output,
                    backend=materialization_backend,
                )

    @staticmethod
    def _get_first_step_read_backend(context: ProcessingContext, vfs_config) -> str:
        """Get read backend for first step based on VFS config and metadata-based auto-detection."""

        # Check if user explicitly configured a read backend
        if vfs_config.read_backend != Backend.AUTO:
            return vfs_config.read_backend.value

        # AUTO mode: Use unified backend detection
        return MaterializationFlagPlanner._detect_backend_for_context(context, fallback_backend=Backend.DISK.value)

    @staticmethod
    def _resolve_materialization_backend(context: ProcessingContext, vfs_config) -> str:
        """Resolve materialization backend, handling AUTO option."""
        # Check if user explicitly configured a materialization backend
        if vfs_config.materialization_backend != MaterializationBackend.AUTO:
            return vfs_config.materialization_backend.value

        # AUTO mode: Use unified backend detection
        return MaterializationFlagPlanner._detect_backend_for_context(context, fallback_backend=MaterializationBackend.DISK.value)

    @staticmethod
    def _last_image_materialization_step(step_plans, step_count: int) -> int | None:
        """Return the last step index whose outputs should seed the output plate."""
        for step_index in range(step_count - 1, -1, -1):
            if MaterializationFlagPlanner._step_materializes_images(
                step_plans[step_index]
            ):
                return step_index
        return None

    @staticmethod
    def _step_materializes_images(step_plan) -> bool:
        """Return whether automatic final materialization should flush images."""
        if not step_plan.artifact_outputs:
            return True
        image_kinds = {ImageArtifactType, ObjectLabelsArtifactType}
        return any(
            output.artifact_type in image_kinds
            and output.materialization is not NO_ARTIFACT_MATERIALIZATION
            for output in step_plan.artifact_outputs.values()
        )

    @staticmethod
    def _detect_backend_for_context(context: ProcessingContext, fallback_backend: str) -> str:
        """Unified backend detection logic for both read and materialization backends."""
        # Use the microscope handler's get_primary_backend method
        # This handles both OpenHCS (metadata-based) and other microscopes (compatibility-based)
        return context.microscope_handler.get_primary_backend(context.input_dir, context.filemanager)
