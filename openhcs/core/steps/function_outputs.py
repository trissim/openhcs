"""Output finalization for FunctionStep execution."""

from __future__ import annotations

import logging
import time

from openhcs.constants.constants import Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.steps.function_artifact_materialization import (
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_io import (
    calculate_zarr_dimensions,
    generate_materialized_paths,
    save_materialized_data,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan


logger = logging.getLogger(__name__)


def finalize_function_step_outputs(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    """Persist images, streams, metadata, and non-image artifacts for one step."""
    _write_memory_outputs_if_needed(context, plan)
    _materialize_images_if_needed(context, plan)
    _stream_outputs(context, plan)
    _write_openhcs_metadata(context, plan)
    _materialize_artifacts(context, plan)


def _write_memory_outputs_if_needed(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    if plan.write_backend == Backend.MEMORY.value:
        return

    memory_paths = plan.get_paths_for_axis(plan.output_dir, Backend.MEMORY.value)
    memory_data = context.filemanager.load_batch(memory_paths, Backend.MEMORY.value)
    n_channels, n_z, n_fields = calculate_zarr_dimensions(
        memory_paths, context.microscope_handler
    )
    row, col = context.microscope_handler.parser.extract_component_coordinates(
        plan.axis_id
    )
    context.filemanager.ensure_directory(plan.output_dir, plan.write_backend)
    context.filemanager.save_batch(
        memory_data,
        memory_paths,
        plan.write_backend,
        chunk_name=plan.axis_id,
        zarr_config=plan.zarr_config,
        n_channels=n_channels,
        n_z=n_z,
        n_fields=n_fields,
        row=row,
        col=col,
        parser_name=context.microscope_handler.parser.__class__.__name__,
        microscope_type=context.microscope_handler.microscope_type,
    )


def _materialize_images_if_needed(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    if not plan.has_materialized_output:
        return

    memory_paths = plan.get_paths_for_axis(plan.output_dir, Backend.MEMORY.value)
    memory_data = context.filemanager.load_batch(memory_paths, Backend.MEMORY.value)
    materialized_paths = generate_materialized_paths(
        memory_paths,
        plan.output_dir,
        plan.materialized_output_dir,
    )

    context.filemanager.ensure_directory(
        plan.materialized_output_dir,
        plan.materialized_backend,
    )
    save_materialized_data(
        context.filemanager,
        memory_data,
        materialized_paths,
        plan.materialized_backend,
        plan.zarr_config,
        context,
        plan.axis_id,
    )
    logger.info(
        "Materialized %s files to %s",
        len(materialized_paths),
        plan.materialized_output_dir,
    )


def _stream_outputs(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    for config_instance in plan.streaming_configs:
        memory_paths = plan.get_paths_for_axis(
            plan.output_dir,
            Backend.MEMORY.value,
        )
        if plan.has_materialized_output:
            streaming_paths = generate_materialized_paths(
                memory_paths,
                plan.output_dir,
                plan.materialized_output_dir,
            )
        else:
            streaming_paths = memory_paths

        streaming_data = context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        kwargs = config_instance.get_streaming_kwargs(context)
        kwargs["source"] = plan.step_name
        context.filemanager.save_batch(
            streaming_data,
            streaming_paths,
            config_instance.backend.value,
            **kwargs,
        )
        time.sleep(0.1)


def _write_openhcs_metadata(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    if plan.write_backend not in [Backend.OMERO_LOCAL.value, Backend.MEMORY.value]:
        from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

        OpenHCSMetadataGenerator(context.filemanager).create_metadata(
            context,
            str(plan.output_dir),
            plan.write_backend,
            is_main=plan.write_backend != Backend.MEMORY.value,
            plate_root=plan.output_plate_root,
            sub_dir=plan.sub_dir,
            results_dir=plan.analysis_results_dir,
        )

    if not plan.has_materialized_output:
        return

    if plan.materialized_backend in [Backend.OMERO_LOCAL.value, Backend.MEMORY.value]:
        return

    from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

    OpenHCSMetadataGenerator(context.filemanager).create_metadata(
        context,
        str(plan.materialized_output_dir),
        plan.materialized_backend,
        is_main=False,
        plate_root=plan.materialized_plate_root,
        sub_dir=plan.materialized_sub_dir,
        results_dir=plan.materialized_analysis_results_dir,
    )


def _materialize_artifacts(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    if not plan.artifact_outputs:
        return

    logger.info(
        "Starting materialization for %s artifact outputs",
        len(plan.artifact_outputs),
    )
    from openhcs.core.pipeline.materialization_flag_planner import (
        MaterializationFlagPlanner,
    )

    materialization_backend = (
        MaterializationFlagPlanner._resolve_materialization_backend(
            context,
            context.get_vfs_config(),
        )
    )
    materialize_artifact_outputs(
        context.filemanager,
        plan,
        materialization_backend,
        context,
    )
    logger.info("Completed artifact materialization")
