"""Output finalization for FunctionStep execution."""

from __future__ import annotations

import logging
import time

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamProducer

from openhcs.constants.constants import Backend, VariableComponents
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.compiled_step_plan import RuntimeArtifactMaterializationPlan
from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectedPayloadItem,
    RuntimeProjectionSourceIdentityRequest,
    RuntimeProjectionSourceIdentityRequirement,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    image_payload_data,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
)
from openhcs.core.steps.function_artifact_materialization import (
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentityAuthority,
    FunctionOutputParserContext,
    FunctionOutputPathAuthority,
)
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.function_io import (
    calculate_zarr_dimensions,
    generate_materialized_paths,
    save_materialized_data,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraAuthority,
    StreamSourceComponentMetadataItems,
)
from openhcs.microscopes.microscope_interfaces import FilenameParser

logger = logging.getLogger(__name__)
StreamPayload = ImagePayloadMetadataInput


def stream_payload_summary(payload: StreamPayload) -> str:
    """Return bounded image payload facts for runtime streaming diagnostics."""
    data = image_payload_data(payload)
    if not isinstance(data, np.ndarray):
        return f"type={type(data).__name__}"

    summary = (
        f"shape={tuple(int(axis) for axis in data.shape)} "
        f"dtype={data.dtype} size={int(data.size)} "
        f"nonzero={int(np.count_nonzero(data))}"
    )
    if not data.size:
        return summary
    return f"{summary} min={data.min()} max={data.max()}"

class ProducedMemoryPathsAuthority:
    """Resolve absolute memory paths produced by the current step execution."""

    @classmethod
    def paths(
        cls,
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> list[str]:
        return [
            cls.memory_path(record, plan)
            for record in step_output_manifest(context).produced_records_for(
                plan
            )
        ]

    @staticmethod
    def memory_path(
        record: ProducedOutputSemantics,
        plan: FunctionStepExecutionPlan,
    ) -> str:
        path = Path(record.output_path)
        if path.is_absolute():
            return str(path)
        return str(plan.output_dir / record.relative_output_path)

def finalize_function_step_outputs(
    context: ProcessingContext,
    plan: FunctionStepExecutionPlan,
) -> None:
    """Persist images, streams, metadata, and non-image artifacts for one step."""
    if not RuntimeProfileLogger.enabled():
        MemoryOutputWriter.write_if_needed(context, plan)
        MaterializedImageOutputWriter.write_if_needed(context, plan)
        StreamOutputsAuthority.stream_outputs(context, plan)
        OpenHCSMetadataWriter.write(context, plan)
        RuntimeArtifactMaterializationAuthority.materialize(context, plan)
        return

    _profile_finalization_phase(
        "finalize_memory_outputs",
        lambda: MemoryOutputWriter.write_if_needed(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_materialized_images",
        lambda: MaterializedImageOutputWriter.write_if_needed(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_stream_outputs",
        lambda: StreamOutputsAuthority.stream_outputs(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_openhcs_metadata",
        lambda: OpenHCSMetadataWriter.write(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_runtime_artifacts",
        lambda: RuntimeArtifactMaterializationAuthority.materialize(context, plan),
        plan,
    )


def _profile_finalization_phase(
    label: str,
    operation: Callable[[], None],
    plan: FunctionStepExecutionPlan,
) -> None:
    started_at = time.perf_counter()
    operation()
    RuntimeProfileLogger.log(
        logger,
        label,
        time.perf_counter() - started_at,
        step=plan.step_index,
        step_name=plan.step_name,
        axis_id=plan.axis_id,
    )

class MemoryOutputWriter:
    """Writes memory-backed step outputs to the configured write backend."""

    @classmethod
    def write_if_needed(
        cls,
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        if plan.write_backend == Backend.MEMORY.value:
            return

        memory_paths = ProducedMemoryPathsAuthority.paths(context, plan)
        if not memory_paths:
            return
        memory_data = context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        n_channels, n_z, n_fields = calculate_zarr_dimensions(
            memory_paths,
            context.microscope_handler,
        )
        parser_context = FunctionOutputParserContext.from_processing_context(
            context
        )
        row, col = parser_context.parser.extract_component_coordinates(
            plan.axis_id
        )
        context.filemanager.ensure_directory(
            plan.output_dir,
            plan.write_backend,
        )
        context.filemanager.save_batch(
            cls.payloads(memory_data, memory_paths, plan),
            memory_paths,
            plan.write_backend,
            chunk_name=plan.axis_id,
            zarr_config=plan.zarr_config,
            n_channels=n_channels,
            n_z=n_z,
            n_fields=n_fields,
            row=row,
            col=col,
            parser_name=parser_context.parser_name,
            microscope_type=parser_context.microscope_type,
        )

    @staticmethod
    def payloads(
        memory_data: list[StreamPayload],
        memory_paths: list[str],
        plan: FunctionStepExecutionPlan,
    ) -> list[StreamPayload]:
        if plan.write_backend == Backend.DISK.value:
            return prepare_disk_image_payloads(memory_data, memory_paths)
        return memory_data

class MaterializedImageOutputWriter:
    """Materializes image outputs for steps configured with materialized output."""

    @staticmethod
    def write_if_needed(
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        if not plan.has_materialized_output:
            return

        memory_paths = ProducedMemoryPathsAuthority.paths(context, plan)
        if not memory_paths:
            return
        memory_data = context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
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

@dataclass(frozen=True, slots=True)
class StreamOutputProjectionRequest:
    """Nominal source of truth for projecting produced outputs into viewer streams."""

    parser: FilenameParser
    payloads: tuple[StreamPayload, ...]
    paths: tuple[str, ...]
    produced_outputs: tuple[ProducedOutputSemantics, ...]
    variable_components: tuple[VariableComponents, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        parser: FilenameParser,
        payloads: list[StreamPayload],
        paths: list[str],
        produced_outputs: tuple[ProducedOutputSemantics, ...],
        variable_components: Sequence[VariableComponents],
    ) -> "StreamOutputProjectionRequest":
        return cls(
            parser=parser,
            payloads=tuple(payloads),
            paths=tuple(paths),
            produced_outputs=produced_outputs,
            variable_components=tuple(variable_components),
        )

    def __post_init__(self) -> None:
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
        if not self.produced_outputs:
            raise ValueError("Streaming requires at least one produced output record.")

    def require_single_producer_identity(self) -> StreamProducerIdentity:
        producer_identity = self.produced_outputs[0].producer_identity
        for produced_output in self.produced_outputs[1:]:
            if produced_output.producer_identity != producer_identity:
                raise ValueError("A viewer stream batch cannot mix producer identities.")
        return producer_identity

    def runtime_projection_request(
        self,
        payload: StreamPayload,
        path: str,
    ) -> RuntimeProjectionSourceIdentityRequest:
        return RuntimeProjectionSourceIdentityRequest(
            value=payload,
            source_description=path,
            variable_components=self.variable_components,
        )

    def for_producer(
        self,
        producer_identity: StreamProducerIdentity,
    ) -> "StreamOutputProjectionRequest":
        payloads: list[StreamPayload] = []
        paths: list[str] = []
        produced_outputs: list[ProducedOutputSemantics] = []
        for payload, path, produced_output in zip(
            self.payloads,
            self.paths,
            self.produced_outputs,
            strict=True,
        ):
            if produced_output.producer_identity == producer_identity:
                payloads.append(payload)
                paths.append(path)
                produced_outputs.append(produced_output)
        return StreamOutputProjectionRequest.from_sequences(
            parser=self.parser,
            payloads=payloads,
            paths=paths,
            produced_outputs=tuple(produced_outputs),
            variable_components=self.variable_components,
        )


@dataclass(frozen=True, slots=True)
class StreamOutputBatch:
    """Projected viewer stream payloads with one authoritative producer identity."""

    data_items: tuple[StreamPayload, ...]
    output_paths: tuple[str, ...]
    source_metadata_items: StreamSourceComponentMetadataItems
    producer: ViewerStreamProducer

    @classmethod
    def from_projection(
        cls,
        request: StreamOutputProjectionRequest,
    ) -> "StreamOutputBatch":
        producer_identity = request.require_single_producer_identity()

        data_items: list[StreamPayload] = []
        output_paths: list[str] = []
        component_metadata: list[SourceComponentMetadata] = []
        for payload, path, produced_output in zip(
            request.payloads,
            request.paths,
            request.produced_outputs,
            strict=True,
        ):
            projected_items = tuple(
                cls.project_item(request.runtime_projection_request(payload, path))
            )
            for projected_item in projected_items:
                stream_path = cls.stream_path_for_projected_item(
                    projected_item,
                    produced_path=path,
                    produced_output=produced_output,
                    parser=request.parser,
                    projected_item_count=len(projected_items),
                )
                source_metadata = projected_item.require_source_component_metadata()
                logger.info(
                    "🔬 STREAM SEND: path=%s components=%s %s",
                    stream_path,
                    dict(source_metadata),
                    stream_payload_summary(projected_item.data),
                )
                data_items.append(projected_item.data)
                output_paths.append(stream_path)
                component_metadata.append(source_metadata)

        return cls(
            data_items=tuple(data_items),
            output_paths=tuple(output_paths),
            source_metadata_items=StreamSourceComponentMetadataItems.from_values(
                component_metadata
            ),
            producer=ViewerStreamProducer.from_identity(producer_identity),
        )

    @classmethod
    def from_projection_by_producer(
        cls,
        request: StreamOutputProjectionRequest,
    ) -> tuple["StreamOutputBatch", ...]:
        producer_identities = tuple(
            dict.fromkeys(
                produced_output.producer_identity
                for produced_output in request.produced_outputs
            )
        )

        return tuple(
            cls.from_projection(request.for_producer(producer_identity))
            for producer_identity in producer_identities
        )

    @property
    def is_empty(self) -> bool:
        return not self.data_items

    @property
    def data_list(self) -> list[StreamPayload]:
        return list(self.data_items)

    @property
    def paths(self) -> list[str]:
        return list(self.output_paths)

    @staticmethod
    def project_item(
        request: RuntimeProjectionSourceIdentityRequest,
    ) -> tuple[RuntimeProjectedPayloadItem, ...]:
        return (
            RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
        ).project_payload_items(request)

    @staticmethod
    def stream_path_for_projected_item(
        projected_item: RuntimeProjectedPayloadItem,
        *,
        produced_path: str,
        produced_output: ProducedOutputSemantics,
        parser: FilenameParser,
        projected_item_count: int,
    ) -> str:
        """Return the stream-visible path for one projected payload item."""
        if projected_item_count <= 1:
            return produced_path
        identity = FunctionOutputIdentityAuthority.identity_from_metadata(
            parser,
            projected_item.metadata,
            fallback_identity_path=produced_path,
        )
        if identity is None:
            return produced_path
        if produced_output.filename_qualifier is not None:
            identity = identity.with_filename_qualifier(
                produced_output.filename_qualifier
            )
        filename = FunctionOutputPathAuthority.filename_for_identity(
            parser,
            identity,
        )
        return str(Path(produced_path).parent / filename)

class StreamOutputsAuthority:
    """Streams step image outputs through viewer backends."""

    @staticmethod
    def stream_outputs(
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        for config_instance in plan.streaming_configs:
            produced_outputs = tuple(
                record
                for record in step_output_manifest(context).produced_records_for(plan)
                if record.streams_as_image_payload
            )
            memory_paths = [
                ProducedMemoryPathsAuthority.memory_path(record, plan)
                for record in produced_outputs
            ]
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
            stream_batches = StreamOutputBatch.from_projection_by_producer(
                StreamOutputProjectionRequest.from_sequences(
                    parser=context.microscope_handler.parser,
                    payloads=streaming_payloads,
                    paths=list(streaming_paths),
                    produced_outputs=produced_outputs,
                    variable_components=plan.variable_components,
                )
            )
            stream_batches = tuple(
                stream_batch
                for stream_batch in stream_batches
                if not stream_batch.is_empty
            )
            if not stream_batches:
                logger.info(
                    "No streamable image outputs for step %s after stack projection.",
                    plan.step_name,
                )
                continue
            viewer_surface = config_instance.streaming_viewer_surface(context)
            for stream_batch in stream_batches:
                stream_backend_kwargs = (
                    StreamComponentMessageExtraAuthority.from_context(
                        viewer_surface,
                        context=context,
                        source_metadata_items=(
                            stream_batch.source_metadata_items
                        ),
                    ).viewer_backend_kwargs(
                        producer=stream_batch.producer,
                    )
                )
                context.filemanager.save_batch(
                    stream_batch.data_list,
                    stream_batch.paths,
                    config_instance.backend.value,
                    **stream_backend_kwargs.to_kwargs(),
                )

class OpenHCSMetadataWriter:
    """Writes OpenHCS metadata sidecars for primary and materialized outputs."""

    @classmethod
    def write(
        cls,
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        cls.write_primary_metadata(context, plan)
        cls.write_materialized_metadata(context, plan)

    @staticmethod
    def write_primary_metadata(
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        if plan.write_backend in [Backend.OMERO_LOCAL.value, Backend.MEMORY.value]:
            return

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

    @staticmethod
    def write_materialized_metadata(
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        if not plan.has_materialized_output:
            return
        if plan.materialized_backend in [
            Backend.OMERO_LOCAL.value,
            Backend.MEMORY.value,
        ]:
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

class RuntimeArtifactMaterializationAuthority:
    """Materializes runtime artifacts for persistent output and streaming."""

    @classmethod
    def materialize(
        cls,
        context: ProcessingContext,
        plan: FunctionStepExecutionPlan,
    ) -> None:
        if not plan.artifact_outputs:
            return
        materialization_plan = plan.runtime_artifact_materialization
        has_persistent_target = materialization_plan.has_persistent_target
        has_streaming_target = bool(plan.streaming_configs)
        if not has_persistent_target and not has_streaming_target:
            logger.info("Skipping runtime artifact materialization and streaming")
            return

        logger.info(
            "Starting materialization for %s artifact outputs",
            len(plan.artifact_outputs),
        )
        materialize_artifact_outputs(
            context.filemanager,
            plan,
            cls.target_plan(materialization_plan),
            context,
        )
        logger.info("Completed artifact materialization")

    @staticmethod
    def target_plan(
        materialization_plan: RuntimeArtifactMaterializationPlan,
    ) -> (
        PersistentArtifactMaterializationTargetPlan
        | StreamingOnlyArtifactMaterializationTargetPlan
    ):
        if not materialization_plan.has_persistent_target:
            logger.info("Skipping persistent runtime artifact materialization")
            return StreamingOnlyArtifactMaterializationTargetPlan()

        return PersistentArtifactMaterializationTargetPlan(
            materialization_plan.require_persistent_backend()
        )
