"""Output finalization for FunctionStep execution."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import Any, Mapping

from openhcs.constants.constants import Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.image_shapes import is_image_stack
from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.runtime_values import image_payload_data, image_payload_metadata
from openhcs.core.steps.function_artifact_materialization import (
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
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
    StreamOutputsAuthority.stream_outputs(context, plan)
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
    payloads = (
        prepare_disk_image_payloads(memory_data, memory_paths)
        if plan.write_backend == Backend.DISK.value
        else memory_data
    )
    context.filemanager.save_batch(
        payloads,
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


class StreamOutputsAuthority:
    """Streams step image outputs through viewer backends."""

    @dataclass(frozen=True, slots=True)
    class StreamItem:
        """One semantically identified image payload sent to a viewer backend."""

        data: Any
        path: str
        component_metadata: Mapping[str, Any] | None

    @classmethod
    def _stream_items_for_payload(
        cls,
        payload: Any,
        path: str,
    ) -> tuple["StreamOutputsAuthority.StreamItem", ...]:
        data = image_payload_data(payload)
        metadata = image_payload_metadata(payload)

        if not is_image_stack(data):
            return (
                cls.StreamItem(
                    data=data,
                    path=path,
                    component_metadata=metadata.source_component_metadata,
                ),
            )

        channel_metadata = metadata.channel_source_component_metadata
        if not channel_metadata or any(item is None for item in channel_metadata):
            raise ValueError(
                "Streaming an OpenHCS image stack requires per-slice "
                f"component metadata; got stack shape {getattr(data, 'shape', None)!r} "
                f"for {path!r}."
            )

        if len(channel_metadata) != data.shape[0]:
            raise ValueError(
                "Streaming image stack metadata cardinality mismatch: "
                f"{len(channel_metadata)} metadata entries for stack shape "
                f"{getattr(data, 'shape', None)!r} at {path!r}."
            )

        return tuple(
            cls.StreamItem(
                data=data[index],
                path=path,
                component_metadata=metadata.for_channel(index).source_component_metadata,
            )
            for index in range(data.shape[0])
        )

    @classmethod
    def _stream_items(
        cls,
        payloads: list[Any],
        paths: list[str],
    ) -> tuple["StreamOutputsAuthority.StreamItem", ...]:
        if len(payloads) != len(paths):
            raise ValueError(
                "Streaming payload/path cardinality mismatch: "
                f"{len(payloads)} payloads for {len(paths)} paths."
            )
        return tuple(
            item
            for payload, path in zip(payloads, paths)
            for item in cls._stream_items_for_payload(payload, path)
        )

    @staticmethod
    def stream_outputs(
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

            streaming_payloads = context.filemanager.load_batch(
                memory_paths,
                Backend.MEMORY.value,
            )
            stream_items = StreamOutputsAuthority._stream_items(
                list(streaming_payloads),
                list(streaming_paths),
            )
            kwargs = config_instance.get_streaming_kwargs(context)
            kwargs["source"] = plan.step_name
            kwargs["component_metadata_by_path"] = tuple(
                item.component_metadata for item in stream_items
            )
            context.filemanager.save_batch(
                [item.data for item in stream_items],
                [item.path for item in stream_items],
                config_instance.backend.value,
                **kwargs,
            )


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
    if (
        not context.global_config.materialize_runtime_artifacts
        and not plan.streaming_configs
    ):
        logger.info("Skipping runtime artifact materialization and streaming")
        return

    logger.info(
        "Starting materialization for %s artifact outputs",
        len(plan.artifact_outputs),
    )
    from openhcs.core.pipeline.materialization_flag_planner import (
        MaterializationFlagPlanner,
    )

    target_plan = StreamingOnlyArtifactMaterializationTargetPlan()
    if context.global_config.materialize_runtime_artifacts:
        target_plan = PersistentArtifactMaterializationTargetPlan(
            MaterializationFlagPlanner._resolve_materialization_backend(
                context,
                context.get_vfs_config(),
            )
        )
    else:
        logger.info("Skipping persistent runtime artifact materialization")

    materialize_artifact_outputs(
        context.filemanager,
        plan,
        target_plan,
        context,
    )
    logger.info("Completed artifact materialization")
