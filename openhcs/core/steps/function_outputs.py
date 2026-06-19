"""Output finalization for FunctionStep execution."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path

from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamBackendKwargs

from openhcs.constants.constants import Backend
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.image_shapes import ArrayShape, is_image_stack
from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataInput,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.steps.function_artifact_materialization import (
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_output_identity import FunctionOutputParserContext
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    StepOutputStreamIdentityAuthority,
    step_output_manifest,
)
from openhcs.core.steps.function_io import (
    calculate_zarr_dimensions,
    generate_materialized_paths,
    save_materialized_data,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.stream_component_semantics import (
    OpenHCSViewerStreamRequestAuthority,
    OpenHCSViewerStreamSourceScope,
    StreamComponentMetadata,
    StreamSourceComponentMetadataItems,
)


logger = logging.getLogger(__name__)
StreamPayload = ImagePayloadMetadataInput


@dataclass(frozen=True, slots=True)
class ProducedMemoryPathsAuthority:
    """Resolve absolute memory paths produced by the current step execution."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def paths(self) -> list[str]:
        return [
            self.memory_path(record)
            for record in step_output_manifest(self.context).produced_records_for(
                self.plan
            )
        ]

    def memory_path(self, record: ProducedOutputSemantics) -> str:
        path = Path(record.output_path)
        if path.is_absolute():
            return str(path)
        return str(self.plan.output_dir / record.relative_output_path)


def finalize_function_step_outputs(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    """Persist images, streams, metadata, and non-image artifacts for one step."""
    FunctionStepOutputFinalization(context, plan).execute()


@dataclass(frozen=True, slots=True)
class FunctionStepOutputFinalization:
    """Nominal output-finalization pipeline for one function step."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def execute(self) -> None:
        MemoryOutputWriter(self.context, self.plan).write_if_needed()
        MaterializedImageOutputWriter(self.context, self.plan).write_if_needed()
        StreamOutputsAuthority.stream_outputs(self.context, self.plan)
        OpenHCSMetadataWriter(self.context, self.plan).write()
        RuntimeArtifactMaterializationAuthority(self.context, self.plan).materialize()


@dataclass(frozen=True, slots=True)
class MemoryOutputWriter:
    """Writes memory-backed step outputs to the configured write backend."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def write_if_needed(self) -> None:
        if self.plan.write_backend == Backend.MEMORY.value:
            return

        memory_paths = ProducedMemoryPathsAuthority(self.context, self.plan).paths()
        if not memory_paths:
            return
        memory_data = self.context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        n_channels, n_z, n_fields = calculate_zarr_dimensions(
            memory_paths,
            self.context.microscope_handler,
        )
        parser_context = FunctionOutputParserContext.from_processing_context(
            self.context
        )
        row, col = parser_context.parser.extract_component_coordinates(
            self.plan.axis_id
        )
        self.context.filemanager.ensure_directory(
            self.plan.output_dir,
            self.plan.write_backend,
        )
        self.context.filemanager.save_batch(
            self.payloads(memory_data, memory_paths),
            memory_paths,
            self.plan.write_backend,
            chunk_name=self.plan.axis_id,
            zarr_config=self.plan.zarr_config,
            n_channels=n_channels,
            n_z=n_z,
            n_fields=n_fields,
            row=row,
            col=col,
            parser_name=parser_context.parser_name,
            microscope_type=parser_context.microscope_type,
        )

    def payloads(
        self,
        memory_data: list[StreamPayload],
        memory_paths: list[str],
    ) -> list[StreamPayload]:
        if self.plan.write_backend == Backend.DISK.value:
            return prepare_disk_image_payloads(memory_data, memory_paths)
        return memory_data


@dataclass(frozen=True, slots=True)
class MaterializedImageOutputWriter:
    """Materializes image outputs for steps configured with materialized output."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def write_if_needed(self) -> None:
        if not self.plan.has_materialized_output:
            return

        memory_paths = ProducedMemoryPathsAuthority(self.context, self.plan).paths()
        if not memory_paths:
            return
        memory_data = self.context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        materialized_paths = generate_materialized_paths(
            memory_paths,
            self.plan.output_dir,
            self.plan.materialized_output_dir,
        )

        self.context.filemanager.ensure_directory(
            self.plan.materialized_output_dir,
            self.plan.materialized_backend,
        )
        save_materialized_data(
            self.context.filemanager,
            memory_data,
            materialized_paths,
            self.plan.materialized_backend,
            self.plan.zarr_config,
            self.context,
            self.plan.axis_id,
        )
        logger.info(
            "Materialized %s files to %s",
            len(materialized_paths),
            self.plan.materialized_output_dir,
        )


@dataclass(frozen=True, slots=True)
class StreamItem:
    """One viewer-backend payload with the source metadata used for layer identity."""

    data: StreamPayload
    path: str
    produced_output: ProducedOutputSemantics
    stream_source_component_metadata: StreamComponentMetadata


@dataclass(frozen=True, slots=True)
class StreamPayloadShape:
    """Nominal shape view for deciding whether a payload can be split safely."""

    payload: StreamPayload

    @property
    def array_shape(self) -> ArrayShape | None:
        return ArrayShape.from_value(self.payload)

    @property
    def diagnostic_shape(self) -> tuple[int, ...] | None:
        array_shape = self.array_shape
        if array_shape is None:
            return None
        return array_shape.shape

    @property
    def leading_axis_length(self) -> int:
        array_shape = self.array_shape
        if array_shape is None:
            raise ValueError("Streaming stack payload has no array shape.")
        return array_shape.shape[0]

    @property
    def is_semantic_stack_shape(self) -> bool:
        return is_image_stack(self.payload)


@dataclass(frozen=True, slots=True)
class StreamStackSliceMetadata:
    """Per-slice metadata authority for semantic stack projection."""

    metadata: tuple[StreamComponentMetadata, ...]

    @classmethod
    def from_payload_metadata(
        cls,
        payload_metadata: ImagePayloadMetadata,
    ) -> "StreamStackSliceMetadata":
        return cls(tuple(payload_metadata.source_image_provenance_planes.component_metadata))

    @property
    def is_complete(self) -> bool:
        return bool(self.metadata) and all(item is not None for item in self.metadata)

    def validate_cardinality(self, payload_shape: StreamPayloadShape, path: str) -> None:
        if len(self.metadata) == payload_shape.leading_axis_length:
            return
        raise ValueError(
            "Streaming image stack metadata cardinality mismatch: "
            f"{len(self.metadata)} metadata entries for stack shape "
            f"{payload_shape.diagnostic_shape!r} at {path!r}."
        )


@dataclass(frozen=True, slots=True)
class StreamPathContext:
    """Path and processing context shared by stream payload authorities."""

    context: ProcessingContext
    path: str


@dataclass(frozen=True, slots=True)
class StreamPayloadItemsRequest(StreamPathContext):
    """Inputs needed to project one runtime payload into viewer stream items."""

    payload: StreamPayload
    produced_output: ProducedOutputSemantics

    @property
    def data(self) -> StreamPayload:
        return image_payload_data(self.payload)

    @property
    def metadata(self) -> ImagePayloadMetadata:
        return image_payload_metadata(self.payload)


@dataclass(frozen=True, slots=True)
class StreamPayloadItemProjection:
    """Projects one runtime payload only when provenance proves slice identity."""

    request: StreamPayloadItemsRequest

    def apply(self) -> tuple[StreamItem, ...]:
        data = self.request.data
        metadata = self.request.metadata
        payload_shape = StreamPayloadShape(data)
        stack_metadata = StreamStackSliceMetadata.from_payload_metadata(metadata)

        if not payload_shape.is_semantic_stack_shape:
            return (
                StreamItem(
                    data=data,
                    path=self.request.path,
                    produced_output=self.request.produced_output,
                    stream_source_component_metadata=(
                        self.request.produced_output.component_metadata
                    ),
                ),
            )

        if not stack_metadata.is_complete:
            logger.warning(
                "Skipping streaming image stack without complete per-slice "
                "component metadata: path=%r shape=%r.",
                self.request.path,
                payload_shape.diagnostic_shape,
            )
            return ()

        stack_metadata.validate_cardinality(payload_shape, self.request.path)
        return tuple(
            StreamItem(
                data=data[index],
                path=self.request.path,
                produced_output=self.request.produced_output,
                stream_source_component_metadata=metadata.for_source_plane(index).source_component_metadata,
            )
            for index in range(payload_shape.leading_axis_length)
        )


@dataclass(frozen=True, slots=True)
class StreamBatchItemProjection:
    """Projects all runtime payloads for one streaming backend call."""

    payloads: list[StreamPayload]
    paths: list[str]
    produced_outputs: tuple[ProducedOutputSemantics, ...]
    context: ProcessingContext

    def apply(self) -> tuple[StreamItem, ...]:
        if len(self.payloads) != len(self.paths):
            raise ValueError(
                "Streaming payload/path cardinality mismatch: "
                f"{len(self.payloads)} payloads for {len(self.paths)} paths."
            )
        if len(self.payloads) != len(self.produced_outputs):
            raise ValueError(
                "Streaming payload/output-record cardinality mismatch: "
                f"{len(self.payloads)} payloads for "
                f"{len(self.produced_outputs)} output records."
            )
        return tuple(
            item
            for payload, path, produced_output in zip(
                self.payloads,
                self.paths,
                self.produced_outputs,
            )
            for item in StreamPayloadItemProjection(
                StreamPayloadItemsRequest(
                    self.context,
                    path,
                    payload,
                    produced_output,
                )
            ).apply()
        )


class StreamOutputsAuthority:
    """Streams step image outputs through viewer backends."""

    @staticmethod
    def producer_identity(plan: FunctionStepExecutionPlan) -> StreamProducerIdentity:
        return StepOutputStreamIdentityAuthority.build(plan)

    @staticmethod
    def stream_outputs(
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        for config_instance in plan.streaming_configs:
            produced_outputs = step_output_manifest(context).produced_records_for(plan)
            memory_paths = ProducedMemoryPathsAuthority(context, plan).paths()
            if not memory_paths:
                logger.info(
                    "No produced image outputs to stream for step %s.",
                    plan.step_name,
                )
                continue
            if plan.has_materialized_output:
                streaming_paths = generate_materialized_paths(
                    memory_paths,
                    plan.output_dir,
                    plan.materialized_output_dir,
                )
            else:
                streaming_paths = memory_paths

            streaming_payloads: list[StreamPayload] = list(
                context.filemanager.load_batch(
                    memory_paths,
                    Backend.MEMORY.value,
                )
            )
            stream_items = StreamBatchItemProjection(
                streaming_payloads,
                list(streaming_paths),
                produced_outputs,
                context,
            ).apply()
            if not stream_items:
                logger.info(
                    "No streamable image outputs for step %s after stack projection.",
                    plan.step_name,
                )
                continue
            source_metadata_items = StreamSourceComponentMetadataItems.from_values(
                item.stream_source_component_metadata for item in stream_items
            )
            viewer_request = OpenHCSViewerStreamRequestAuthority.from_source_scope(
                OpenHCSViewerStreamSourceScope.from_viewer_surface(
                    config_instance.streaming_viewer_surface(context),
                    context=context,
                    producer_identity=produced_outputs[0].producer_identity,
                    source_metadata_items=source_metadata_items,
                )
            )
            kwargs = ViewerStreamBackendKwargs(viewer_request).to_kwargs()
            context.filemanager.save_batch(
                [item.data for item in stream_items],
                [item.path for item in stream_items],
                config_instance.backend.value,
                **kwargs,
            )


@dataclass(frozen=True, slots=True)
class OpenHCSMetadataWriter:
    """Writes OpenHCS metadata sidecars for primary and materialized outputs."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def write(self) -> None:
        self.write_primary_metadata()
        self.write_materialized_metadata()

    def write_primary_metadata(self) -> None:
        if self.plan.write_backend in [Backend.OMERO_LOCAL.value, Backend.MEMORY.value]:
            return

        from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

        OpenHCSMetadataGenerator(self.context.filemanager).create_metadata(
            self.context,
            str(self.plan.output_dir),
            self.plan.write_backend,
            is_main=self.plan.write_backend != Backend.MEMORY.value,
            plate_root=self.plan.output_plate_root,
            sub_dir=self.plan.sub_dir,
            results_dir=self.plan.analysis_results_dir,
        )

    def write_materialized_metadata(self) -> None:
        if not self.plan.has_materialized_output:
            return
        if self.plan.materialized_backend in [
            Backend.OMERO_LOCAL.value,
            Backend.MEMORY.value,
        ]:
            return

        from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

        OpenHCSMetadataGenerator(self.context.filemanager).create_metadata(
            self.context,
            str(self.plan.materialized_output_dir),
            self.plan.materialized_backend,
            is_main=False,
            plate_root=self.plan.materialized_plate_root,
            sub_dir=self.plan.materialized_sub_dir,
            results_dir=self.plan.materialized_analysis_results_dir,
        )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactMaterializationRequest:
    """Policy decision for runtime artifact materialization."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    @property
    def has_artifact_outputs(self) -> bool:
        return bool(self.plan.artifact_outputs)

    @property
    def has_streaming_target(self) -> bool:
        return bool(self.plan.streaming_configs)

    @property
    def has_persistent_target(self) -> bool:
        return bool(self.context.global_config.materialize_runtime_artifacts)

    @property
    def should_materialize(self) -> bool:
        return self.has_artifact_outputs and (
            self.has_persistent_target or self.has_streaming_target
        )

    @property
    def skip_reason(self) -> str | None:
        if not self.has_artifact_outputs:
            return None
        if not self.has_persistent_target and not self.has_streaming_target:
            return "Skipping runtime artifact materialization and streaming"
        return None

    @property
    def target_plan(
        self,
    ) -> (
        PersistentArtifactMaterializationTargetPlan
        | StreamingOnlyArtifactMaterializationTargetPlan
    ):
        if not self.has_persistent_target:
            logger.info("Skipping persistent runtime artifact materialization")
            return StreamingOnlyArtifactMaterializationTargetPlan()

        from openhcs.core.pipeline.materialization_flag_planner import (
            MaterializationFlagPlanner,
        )

        return PersistentArtifactMaterializationTargetPlan(
            MaterializationFlagPlanner._resolve_materialization_backend(
                self.context,
                self.context.get_vfs_config(),
            )
        )


@dataclass(frozen=True, slots=True)
class RuntimeArtifactMaterializationAuthority:
    """Materializes runtime artifacts for persistent output and streaming."""

    context: ProcessingContext
    plan: FunctionStepExecutionPlan

    def materialize(self) -> None:
        request = RuntimeArtifactMaterializationRequest(self.context, self.plan)
        if not request.should_materialize:
            if request.skip_reason is not None:
                logger.info(request.skip_reason)
            return

        logger.info(
            "Starting materialization for %s artifact outputs",
            len(self.plan.artifact_outputs),
        )
        materialize_artifact_outputs(
            self.context.filemanager,
            self.plan,
            request.target_plan,
            self.context,
        )
        logger.info("Completed artifact materialization")
